import logging
import math
import os
import warnings
from transformers.modeling_bert import BertModel, BertPreTrainedModel
from torch import nn
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Dict, Tuple, Optional, Union


def neginf(dtype: torch.dtype) -> float:
    """
    Return a representable finite number near -inf for a dtype.
    """
    NEAR_INF = 1e20
    NEAR_INF_FP16 = 65504
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


class BasicAttention(nn.Module):
    """
    Implements simple/classical attention.
    """

    def __init__(
            self,
            dim: int = 1,
            attn: str = 'cosine',
            residual: bool = False,
            get_weights: bool = True,
    ):
        super().__init__()
        if attn == 'cosine':
            self.cosine = nn.CosineSimilarity(dim=dim)
        self.attn = attn
        self.dim = dim
        self.get_weights = get_weights
        self.residual = residual

    def forward(
            self,
            xs: torch.Tensor,
            ys: torch.Tensor,
            mask_ys: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute attention.

        Attend over ys with query xs to obtain weights, then apply weights to
        values (ys if yalues is None)

        Args:
            xs: B x query_len x dim (queries)
            ys: B x key_len x dim (keys)
            mask_ys: B x key_len (mask)
            values: B x value_len x dim (values); if None, default to ys
        """
        bsz = xs.size(0)
        y_len = ys.size(1)
        x_len = xs.size(1)
        if self.attn == 'cosine':
            l1 = self.cosine(xs, ys).unsqueeze(self.dim - 1)
        else:
            l1 = torch.bmm(xs, ys.transpose(1, 2))
            if self.attn == 'sqrt':
                d_k = ys.size(-1)
                l1 = l1 / math.sqrt(d_k)
        if mask_ys is not None:
            attn_mask = (mask_ys == 0).view(bsz, 1, y_len)
            attn_mask = attn_mask.repeat(1, x_len, 1)
            l1.masked_fill_(attn_mask, neginf(l1.dtype))
        l2 = F.softmax(l1, dim=self.dim, dtype=torch.float).type_as(l1)
        if values is None:
            values = ys
        lhs_emb = torch.bmm(l2, values)

        # # add back the query
        if self.residual:
            lhs_emb = lhs_emb.add(xs)

        if self.get_weights:
            return lhs_emb.squeeze(self.dim - 1), l2
        else:
            return lhs_emb.squeeze(self.dim - 1)


class PolyBasicAttention(BasicAttention):
    """
    Override basic attention to account for edge case for polyencoder.
    """

    def __init__(self, poly_type, n_codes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poly_type = poly_type
        self.n_codes = n_codes

    def forward(self, *args, **kwargs):
        """
        Forward pass.

        Account for accidental dimensionality reduction when num_codes is 1 and the
        polyencoder type is 'codes'
        """
        lhs_emb = super().forward(*args, **kwargs)
        if self.poly_type == 'codes' and self.n_codes == 1 and len(lhs_emb.shape) == 2:
            lhs_emb = lhs_emb.unsqueeze(self.dim - 1)
        return lhs_emb


class Polyencoder_v1(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.type = "codes"
        self.n_codes = 360  # 64
        self.codes_attention_type = "basic"
        self.codes_attention_num_heads = 4  # In case codes-attention-type is multihead, specify the number of heads
        self.attention_type = "basic"
        self.attention_num_heads = 4  # 当 In case poly-attention-type（即attention_type） is multihead, specify the number of heads
        embed_dim = 768

        # In case it's a polyencoder with code.
        if self.type == "codes":
            codes = torch.empty(self.n_codes, embed_dim)
            codes = torch.nn.init.uniform_(codes)
            self.codes = torch.nn.Parameter(codes)
            if self.codes_attention_type == "basic":
                self.code_attention = PolyBasicAttention(
                    self.type, self.n_codes, dim=2, attn='basic', get_weights=False
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # The final attention (the one that takes the candidate as key)
        if self.attention_type == "basic":
            self.attention = PolyBasicAttention(
                self.type,
                self.n_codes,
                dim=2,
                attn=self.attention_type,
                get_weights=False,
            )
        else:
            raise NotImplementedError

    def attend(self, attention_layer, queries, keys, values, mask):
        """
        Apply attention.

        :param attention_layer:
            nn.Module attention layer to use for the attention
        :param queries:
            the queries for attention
        :param keys:
            the keys for attention
        :param values:
            the values for attention
        :param mask:
            mask for the attention keys

        :return:
            the result of applying attention to the values, with weights computed
            wrt to the queries and keys.
        """
        if keys is None:
            keys = values
        if isinstance(attention_layer, PolyBasicAttention):
            return attention_layer(queries, keys, mask_ys=mask, values=values)
        else:
            raise Exception('Unrecognized type of attention')

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,
            r_input_ids=None,
            r_attention_mask=None,
            r_token_type_ids=None,
            labels=None,
            train_mode=True,
    ):
        # 先编码
        # context
        context_outputs = self.bert(c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        context_outputs = context_outputs[0]  # (batch_size, context_len, emb_size)

        # mask类型要转变下
        c_attention_mask = c_attention_mask.float()
        r_attention_mask = r_attention_mask.float()

        if train_mode:
            # response
            response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
            response_outputs = response_outputs[0]  # train: (batch_size, response_len, emb_size)

            # polyencoder
            # (1) 先压缩repsonse 和 context
            # 用 mean 操作压缩response
            divisor = r_attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
            response_outputs = response_outputs.sum(dim=1) / divisor

            batch_size, emb_size = response_outputs.shape
            num_candidates = batch_size
            response_outputs = response_outputs.unsqueeze(0).repeat(num_candidates, 1, 1)  # (bs, num_candidates, emb)

            # 用codes压缩context
            assert self.type == "codes"
            ctxt_rep = self.attend(self.code_attention, queries=self.codes.repeat(batch_size, 1, 1),
                                   keys=context_outputs, values=context_outputs,
                                   mask=c_attention_mask)  # (batch_size, n_codes, emb)
            ctxt_rep_mask = ctxt_rep.new_ones(batch_size, self.n_codes).byte()  # (batch_size, n_codes)

            # (2) context 和 repsonse 交互
            ctxt_final_rep = self.attend(self.attention, response_outputs, ctxt_rep, ctxt_rep,
                                         ctxt_rep_mask)  # (bs, num_candidates, emb)
            scores = torch.sum(ctxt_final_rep * response_outputs, 2)  # (bs, num_candidates)

            # (3) 计算loss
            # 计算loss
            targets = torch.arange(batch_size, dtype=torch.long, device=response_outputs.device)
            loss = F.cross_entropy(scores, targets)

        else:
            # 验证模式
            # response
            # 这里的response已经是一个列表了,只需要改变维度 不用扩充
            batch_size, num_candidates, max_response_len = r_input_ids.shape
            r_input_ids = r_input_ids.view(-1, max_response_len)
            r_attention_mask = r_attention_mask.view(-1, max_response_len)
            r_token_type_ids = r_token_type_ids.view(-1, max_response_len)
            response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
            response_outputs = response_outputs[0]  # (batch_size*num_candidates, response_len, emb_size)

            # polyencoder
            # (1) 先压缩repsonse 和 context
            # 用 mean 操作压缩response
            divisor = r_attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
            response_outputs = response_outputs.sum(dim=1) / divisor
            emb_size = response_outputs.size(-1)
            response_outputs = response_outputs.reshape(batch_size, num_candidates,
                                                        emb_size)  # (bs, num_candidates, emb)

            # 用codes压缩context
            assert self.type == "codes"
            ctxt_rep = self.attend(self.code_attention, queries=self.codes.repeat(batch_size, 1, 1),
                                   keys=context_outputs, values=context_outputs,
                                   mask=c_attention_mask)  # (batch_size, n_codes, emb)
            ctxt_rep_mask = ctxt_rep.new_ones(batch_size, self.n_codes).byte()  # (batch_size, n_codes)

            # (2) context 和 repsonse交互
            ctxt_final_rep = self.attend(self.attention, response_outputs, ctxt_rep, ctxt_rep,
                                         ctxt_rep_mask)  # (bs, num_candidates, emb)
            scores = torch.sum(ctxt_final_rep * response_outputs, 2)  # (bs, num_candidates)
            loss = None

        return scores, loss


class Biencoder_v1(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,
            r_input_ids=None,
            r_attention_mask=None,
            r_token_type_ids=None,
            labels=None,
            train_mode=True,
    ):
        # 先编码
        # context
        context_outputs = self.bert(c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        context_outputs = context_outputs[0]  # (batch_size, context_len, emb_size)

        # mask类型要转变下
        c_attention_mask = c_attention_mask.float()
        r_attention_mask = r_attention_mask.float()

        if train_mode:
            # response
            response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
            response_outputs = response_outputs[0]  # train: (batch_size, response_len, emb_size)

            # biencoder
            # (1) 先压缩 repsonse 和 context
            # 压缩response 取cls
            response_outputs = response_outputs[:, 0, :]

            batch_size, emb_size = response_outputs.shape
            num_candidates = batch_size
            response_outputs = response_outputs.unsqueeze(0).repeat(num_candidates, 1, 1)  # (bs, num_candidates, emb)

            # 压缩 context 取cls
            context_outputs = context_outputs[:, 0, :]
            context_outputs = context_outputs.unsqueeze(1).repeat(1, num_candidates, 1)

            scores = torch.sum(context_outputs * response_outputs, 2)  # (bs, num_candidates)

            # (3) 计算loss
            # 计算loss
            targets = torch.arange(batch_size, dtype=torch.long, device=response_outputs.device)
            loss = F.cross_entropy(scores, targets)

        else:
            # 验证模式
            # response
            # 这里的response已经是一个列表了,只需要改变维度 不用扩充
            batch_size, num_candidates, max_response_len = r_input_ids.shape
            r_input_ids = r_input_ids.view(-1, max_response_len)
            r_attention_mask = r_attention_mask.view(-1, max_response_len)
            r_token_type_ids = r_token_type_ids.view(-1, max_response_len)
            response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
            response_outputs = response_outputs[0]  # (batch_size*num_candidates, response_len, emb_size)

            # biencoder
            # (1) 先压缩 repsonse 和 context
            # 压缩response 取cls
            response_outputs = response_outputs[:, 0, :]
            emb_size = response_outputs.size(-1)
            response_outputs = response_outputs.reshape(batch_size, num_candidates,
                                                        emb_size)  # (bs, num_candidates, emb)

            # 压缩 context 取cls
            context_outputs = context_outputs[:, 0, :]
            context_outputs = context_outputs.unsqueeze(1).repeat(1, num_candidates, 1)

            scores = torch.sum(context_outputs * response_outputs, 2)  # (bs, num_candidates)
            loss = None

        return scores, loss
