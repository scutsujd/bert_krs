import logging
import math
import os
import warnings
from transformers.modeling_bert import BertModel, BertPreTrainedModel
from torch import nn
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss, MSELoss


def match(x, y, x_mask, y_mask):
    # x: (batch_size, m, hidden_size)
    # y: (batch_size, n, hidden_size)
    # x_mask: (batch_size, m)
    # y_mask: (batch_size, n)
    assert x.dim() == 3 and y.dim() == 3
    assert x_mask.dim() == 2 and y_mask.dim() == 2
    assert x_mask.shape == x.shape[:2] and y_mask.shape == y.shape[:2]

    attn_mask = torch.bmm(x_mask.unsqueeze(-1), y_mask.unsqueeze(1))  # (batch_size, m, n)
    attn = torch.bmm(x, y.transpose(1, 2))  # (batch_size, m, n)

    x_to_y = torch.softmax(attn * attn_mask + (-1e6) * (1 - attn_mask), dim=2)  # (batch_size, m, n)
    y_to_x = torch.softmax(attn * attn_mask + (-1e6) * (1 - attn_mask), dim=1).transpose(1, 2)  # # (batch_size, n, m)

    x_attended = torch.bmm(x_to_y, y)  # (batch_size, m, hidden_size)
    y_attended = torch.bmm(y_to_x, x)  # (batch_size, n, hidden_size)

    return x_attended, y_attended


def aggregate(aggregation_method, x, x_mask):
    # x: (batch_size, seq_len, emb_size)
    # x_mask: (batch_size, seq_len)
    assert x.dim() == 3 and x_mask.dim() == 2
    assert x.shape[:2] == x_mask.shape
    # batch_size, seq_len, emb_size = x.shape

    if aggregation_method == "mean":
        return (x * x_mask.unsqueeze(-1)).sum(dim=1) / x_mask.sum(dim=-1, keepdim=True).clamp(
            min=1)  # (batch_size, emb_size)

    if aggregation_method == "max":
        return x.masked_fill(x_mask.unsqueeze(-1) == 0, -1e9).max(dim=1)[0]  # (batch_size, emb_size)

    if aggregation_method == "mean_max":
        return torch.cat([(x * x_mask.unsqueeze(-1)).sum(dim=1) / x_mask.sum(dim=-1, keepdim=True).clamp(min=1), \
                          x.masked_fill(x_mask.unsqueeze(-1) == 0, -1e9).max(dim=1)[0]],
                         dim=-1)  # (batch_size, 2*emb_size)


class BertForPersonaResponseSelection(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        # self.init_weights()

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,
            r_input_ids=None,
            r_attention_mask=None,
            r_token_type_ids=None,
            p_input_ids=None,
            p_attention_mask=None,
            p_token_type_ids=None,
            labels=None,
            train_mode=True,
            aggregate_mode="mean"  # mean / max / mean_max
    ):
        # context
        context_outputs = self.bert(c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        context_outputs = context_outputs[0]  # (batch_size, context_len, emb_size)

        # response
        response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
        response_outputs = response_outputs[0]  # (batch_size, response_len, emb_size)

        # persona
        _, max_persona_num, max_persona_len = p_input_ids.shape
        p_input_ids = p_input_ids.view(-1, max_persona_len)
        p_attention_mask = p_attention_mask.view(-1, max_persona_len)
        p_token_type_ids = p_token_type_ids.view(-1, max_persona_len)
        persona_outputs = self.bert(p_input_ids, attention_mask=p_attention_mask, token_type_ids=p_token_type_ids)
        persona_outputs = persona_outputs[0]  # (batch_size*max_persona_num, max_persona_len, emb_size)
        # 转维度
        emb_size = persona_outputs.shape[-1]
        persona_outputs = persona_outputs.view(-1, max_persona_num, max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num, max_persona_len)
        # persona 拼接起来
        persona_outputs = persona_outputs.view(-1, max_persona_num * max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num * max_persona_len)

        assert context_outputs.dim() == 3 and response_outputs.dim() == 3 and persona_outputs.dim() == 3
        assert c_attention_mask.dim() == 2 and r_attention_mask.dim() == 2 and p_attention_mask.dim() == 2

        # mask类型要转变下
        c_attention_mask = c_attention_mask.float()
        r_attention_mask = r_attention_mask.float()
        p_attention_mask = p_attention_mask.float()

        if train_mode:
            num_candidiates = context_outputs.shape[0]
            batch_size = num_candidiates
            # 先扩充 context和 personas
            context_outputs = context_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, context_len, emb_size)
            c_attention_mask = c_attention_mask.repeat_interleave(num_candidiates, dim=0)
            persona_outputs = persona_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, max_persona_num*num_persona_len, emb_size)
            p_attention_mask = p_attention_mask.repeat_interleave(num_candidiates, dim=0)

            # 看清楚response扩展的维度是不一样的
            _, response_len, emb_size = response_outputs.shape
            response_outputs = response_outputs.unsqueeze(0).repeat(num_candidiates, 1, 1, 1).reshape(-1, response_len,
                                                                                                      emb_size)
            r_attention_mask = r_attention_mask.unsqueeze(0).repeat(num_candidiates, 1, 1).reshape(-1, response_len)

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (2) personas 和 response交互
            p_attend_r, r_attend_p = match(persona_outputs, response_outputs, p_attention_mask, r_attention_mask)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
            p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            r_final = torch.cat([r_mean_max_c, r_mean_max_p], dim=-1)  # (batch_size*num_candidates,2*emb_size)
            c_p_final = torch.cat([c_mean_max, p_mean_max], dim=-1)  # (batch_size*num_candidates,2*emb_size)
            logits = torch.bmm(r_final.unsqueeze(1), c_p_final.unsqueeze(-1)).reshape(batch_size, num_candidiates)

            # 计算loss
            targets = torch.arange(batch_size, dtype=torch.long, device=context_outputs.device)
            loss = F.cross_entropy(logits, targets)

        else:
            # 验证模式
            # 一般设置batch_size为20，因为是从20个选一个出来，而且第一个是正样本，只拿第一个就可以了
            batch_size = context_outputs.shape[0]

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (2) personas 和 response交互
            p_attend_r, r_attend_p = match(persona_outputs, response_outputs, p_attention_mask, r_attention_mask)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
            p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            r_final = torch.cat([r_mean_max_c, r_mean_max_p], dim=-1)  # (batch_size,2*emb_size)
            c_p_final = torch.cat([c_mean_max, p_mean_max], dim=-1)  # (batch_size,2*emb_size)

            logits = torch.bmm(r_final.unsqueeze(1), c_p_final.unsqueeze(-1)).squeeze(
                2)  # (batch_size, 1 , 1) -> (batch_size, 1)
            loss = None

        return logits, loss


class BertForPersonaResponseSelection_V3(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,
            r_input_ids=None,
            r_attention_mask=None,
            r_token_type_ids=None,
            p_input_ids=None,
            p_attention_mask=None,
            p_token_type_ids=None,
            labels=None,
            train_mode=True,
            aggregate_mode="mean"  # mean / max / mean_max
    ):
        # context
        context_outputs = self.bert(c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        context_outputs = context_outputs[0]  # (batch_size, context_len, emb_size)

        # persona
        _, max_persona_num, max_persona_len = p_input_ids.shape
        p_input_ids = p_input_ids.view(-1, max_persona_len)
        p_attention_mask = p_attention_mask.view(-1, max_persona_len)
        p_token_type_ids = p_token_type_ids.view(-1, max_persona_len)
        persona_outputs = self.bert(p_input_ids, attention_mask=p_attention_mask, token_type_ids=p_token_type_ids)
        persona_outputs = persona_outputs[0]  # (batch_size*max_persona_num, max_persona_len, emb_size)
        # 转维度
        emb_size = persona_outputs.shape[-1]
        persona_outputs = persona_outputs.view(-1, max_persona_num, max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num, max_persona_len)

        # mask类型要转变下
        c_attention_mask = c_attention_mask.float()
        r_attention_mask = r_attention_mask.float()
        p_attention_mask = p_attention_mask.float()

        # (1) persona拼接起来然后 attend context
        # 加了这个做法后效果并没有太大差别
        # concat_persona_outputs = persona_outputs.view(-1, max_persona_num * max_persona_len, emb_size)
        # concat_persona_mask = p_attention_mask.view(-1, max_persona_num * max_persona_len)
        # new_c_attend_concat_p, _ = match(context_outputs, concat_persona_outputs, c_attention_mask, concat_persona_mask)
        # new_c_attend_concat_p = self.linear(new_c_attend_concat_p)
        # new_c_attend_concat_p = torch.relu(new_c_attend_concat_p)
        # new_c_attend_concat_p = new_c_attend_concat_p + context_outputs

        # (2) context 和 每个persona 交互
        new_p_list = []
        new_c_list = []
        for i in range(max_persona_num):
            one_p = persona_outputs[:, i]
            one_p_mask = p_attention_mask[:, i]
            one_p_attend_c, one_c_attend_p = match(one_p, context_outputs, one_p_mask, c_attention_mask)
            one_p_attend_c = self.linear1(one_p_attend_c)
            one_p_attend_c = torch.relu(one_p_attend_c)
            one_p_attend_c = one_p_attend_c + one_p
            new_p_list.append(one_p_attend_c)

            one_c_attend_p = self.linear2(one_c_attend_p)
            one_c_attend_p = torch.relu(one_c_attend_p)
            new_c_list.append(one_c_attend_p)

        persona_outputs = torch.stack(new_p_list, dim=1)  # (batch_size, max_persona_num, max_persona_len, emb_size)
        assert persona_outputs.shape[1] == max_persona_num and persona_outputs.shape[2] == max_persona_len
        # persona 拼接起来
        persona_outputs = persona_outputs.view(-1, max_persona_num * max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num * max_persona_len)
        # context 求平均 记得+残差
        new_c_outputs = torch.stack(new_c_list, dim=1)  # (batch_size, max_persona_num, context_len, emb_size)
        new_c_outputs = torch.mean(new_c_outputs, dim=1)  # (batch_size, context_len, emb_size)
        context_outputs = new_c_outputs + context_outputs

        assert context_outputs.dim() == 3 and persona_outputs.dim() == 3
        assert c_attention_mask.dim() == 2 and p_attention_mask.dim() == 2

        if train_mode:
            # response
            response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
            response_outputs = response_outputs[0]  # train: (batch_size, response_len, emb_size)

            num_candidiates = context_outputs.shape[0]
            batch_size = num_candidiates
            # 先扩充 context和 personas
            context_outputs = context_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, context_len, emb_size)
            c_attention_mask = c_attention_mask.repeat_interleave(num_candidiates, dim=0)
            persona_outputs = persona_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, max_persona_num*num_persona_len, emb_size)
            p_attention_mask = p_attention_mask.repeat_interleave(num_candidiates, dim=0)

            # 看清楚response扩展的维度是不一样的
            _, response_len, emb_size = response_outputs.shape
            response_outputs = response_outputs.unsqueeze(0).repeat(num_candidiates, 1, 1, 1).reshape(-1, response_len,
                                                                                                      emb_size)
            r_attention_mask = r_attention_mask.unsqueeze(0).repeat(num_candidiates, 1, 1).reshape(-1, response_len)

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (2) personas 和 response交互
            p_attend_r, r_attend_p = match(persona_outputs, response_outputs, p_attention_mask, r_attention_mask)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
            p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            r_final = torch.cat([r_mean_max_c, r_mean_max_p], dim=-1)  # (batch_size*num_candidates,2*emb_size)
            c_p_final = torch.cat([c_mean_max, p_mean_max], dim=-1)  # (batch_size*num_candidates,2*emb_size)
            logits = torch.bmm(r_final.unsqueeze(1), c_p_final.unsqueeze(-1)).reshape(batch_size, num_candidiates)

            # 计算loss
            targets = torch.arange(batch_size, dtype=torch.long, device=context_outputs.device)
            loss = F.cross_entropy(logits, targets)

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

            # 扩充 context和 personas
            context_outputs = context_outputs.repeat_interleave(num_candidates,
                                                                dim=0)  # (batch_size*num_candidates, context_len, emb_size)
            c_attention_mask = c_attention_mask.repeat_interleave(num_candidates, dim=0)
            persona_outputs = persona_outputs.repeat_interleave(num_candidates,
                                                                dim=0)  # (batch_size*num_candidates, max_persona_num*num_persona_len, emb_size)
            p_attention_mask = p_attention_mask.repeat_interleave(num_candidates, dim=0)

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (2) personas 和 response交互
            p_attend_r, r_attend_p = match(persona_outputs, response_outputs, p_attention_mask, r_attention_mask)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
            p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            r_final = torch.cat([r_mean_max_c, r_mean_max_p], dim=-1)  # (batch_size*num_candidates,2*emb_size)
            c_p_final = torch.cat([c_mean_max, p_mean_max], dim=-1)  # (batch_size*num_candidates,2*emb_size)
            logits = torch.bmm(r_final.unsqueeze(1), c_p_final.unsqueeze(-1)).reshape(batch_size, num_candidates)
            loss = None

        return logits, loss


class BertForPersonaResponseSelection_V3_nocontext(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,
            r_input_ids=None,
            r_attention_mask=None,
            r_token_type_ids=None,
            p_input_ids=None,
            p_attention_mask=None,
            p_token_type_ids=None,
            labels=None,
            train_mode=True,
            aggregate_mode="mean"  # mean / max / mean_max
    ):
        # persona
        _, max_persona_num, max_persona_len = p_input_ids.shape
        p_input_ids = p_input_ids.view(-1, max_persona_len)
        p_attention_mask = p_attention_mask.view(-1, max_persona_len)
        p_token_type_ids = p_token_type_ids.view(-1, max_persona_len)
        persona_outputs = self.bert(p_input_ids, attention_mask=p_attention_mask, token_type_ids=p_token_type_ids)
        persona_outputs = persona_outputs[0]  # (batch_size*max_persona_num, max_persona_len, emb_size)
        # 转维度
        emb_size = persona_outputs.shape[-1]
        persona_outputs = persona_outputs.view(-1, max_persona_num, max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num, max_persona_len)

        # mask类型要转变下
        c_attention_mask = c_attention_mask.float()
        r_attention_mask = r_attention_mask.float()
        p_attention_mask = p_attention_mask.float()

        # persona 拼接起来
        persona_outputs = persona_outputs.view(-1, max_persona_num * max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num * max_persona_len)

        assert persona_outputs.dim() == 3
        assert p_attention_mask.dim() == 2

        if train_mode:
            # response
            response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
            response_outputs = response_outputs[0]  # train: (batch_size, response_len, emb_size)

            num_candidiates = persona_outputs.shape[0]
            batch_size = num_candidiates
            # 先扩充 personas
            persona_outputs = persona_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, max_persona_num*num_persona_len, emb_size)
            p_attention_mask = p_attention_mask.repeat_interleave(num_candidiates, dim=0)

            # 看清楚response扩展的维度是不一样的
            _, response_len, emb_size = response_outputs.shape
            response_outputs = response_outputs.unsqueeze(0).repeat(num_candidiates, 1, 1, 1).reshape(-1, response_len,
                                                                                                      emb_size)
            r_attention_mask = r_attention_mask.unsqueeze(0).repeat(num_candidiates, 1, 1).reshape(-1, response_len)

            # (2) personas 和 response交互
            p_attend_r, r_attend_p = match(persona_outputs, response_outputs, p_attention_mask, r_attention_mask)

            # (3) aggregate 为了减少维度 先用mean
            p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            logits = torch.bmm(r_mean_max_p.unsqueeze(1), p_mean_max.unsqueeze(-1)).reshape(batch_size, num_candidiates)

            # 计算loss
            targets = torch.arange(batch_size, dtype=torch.long, device=persona_outputs.device)
            loss = F.cross_entropy(logits, targets)

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

            # 扩充 context和 personas
            persona_outputs = persona_outputs.repeat_interleave(num_candidates,
                                                                dim=0)  # (batch_size*num_candidates, max_persona_num*num_persona_len, emb_size)
            p_attention_mask = p_attention_mask.repeat_interleave(num_candidates, dim=0)

            # (2) personas 和 response交互
            p_attend_r, r_attend_p = match(persona_outputs, response_outputs, p_attention_mask, r_attention_mask)

            # (3) aggregate 为了减少维度 先用mean
            p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            logits = torch.bmm(r_mean_max_p.unsqueeze(1), p_mean_max.unsqueeze(-1)).reshape(batch_size, num_candidates)
            loss = None

        return logits, loss


class BertForPersonaResponseSelection_V3_nopersona(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,
            r_input_ids=None,
            r_attention_mask=None,
            r_token_type_ids=None,
            p_input_ids=None,
            p_attention_mask=None,
            p_token_type_ids=None,
            labels=None,
            train_mode=True,
            aggregate_mode="mean"  # mean / max / mean_max
    ):
        # context
        context_outputs = self.bert(c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        context_outputs = context_outputs[0]  # (batch_size, context_len, emb_size)

        # mask类型要转变下
        c_attention_mask = c_attention_mask.float()
        r_attention_mask = r_attention_mask.float()
        p_attention_mask = p_attention_mask.float()

        assert context_outputs.dim() == 3
        assert c_attention_mask.dim() == 2

        if train_mode:
            # response
            response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
            response_outputs = response_outputs[0]  # train: (batch_size, response_len, emb_size)

            num_candidiates = context_outputs.shape[0]
            batch_size = num_candidiates
            # 先扩充 context和 personas
            context_outputs = context_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, context_len, emb_size)
            c_attention_mask = c_attention_mask.repeat_interleave(num_candidiates, dim=0)

            # 看清楚response扩展的维度是不一样的
            _, response_len, emb_size = response_outputs.shape
            response_outputs = response_outputs.unsqueeze(0).repeat(num_candidiates, 1, 1, 1).reshape(-1, response_len,
                                                                                                      emb_size)
            r_attention_mask = r_attention_mask.unsqueeze(0).repeat(num_candidiates, 1, 1).reshape(-1, response_len)

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)

            logits = torch.bmm(r_mean_max_c.unsqueeze(1), c_mean_max.unsqueeze(-1)).reshape(batch_size, num_candidiates)

            # 计算loss
            targets = torch.arange(batch_size, dtype=torch.long, device=context_outputs.device)
            loss = F.cross_entropy(logits, targets)

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

            # 扩充 context和 personas
            context_outputs = context_outputs.repeat_interleave(num_candidates,
                                                                dim=0)  # (batch_size*num_candidates, context_len, emb_size)
            c_attention_mask = c_attention_mask.repeat_interleave(num_candidates, dim=0)

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)

            logits = torch.bmm(r_mean_max_c.unsqueeze(1), c_mean_max.unsqueeze(-1)).reshape(batch_size, num_candidates)
            loss = None

        return logits, loss


class BertForPersonaResponseSelection_V2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        # self.init_weights()

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,
            r_input_ids=None,
            r_attention_mask=None,
            r_token_type_ids=None,
            p_input_ids=None,
            p_attention_mask=None,
            p_token_type_ids=None,
            labels=None,
            train_mode=True,
            aggregate_mode="mean"  # mean / max / mean_max
    ):
        # context
        context_outputs = self.bert(c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        context_outputs = context_outputs[0]  # (batch_size, context_len, emb_size)

        # response
        response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
        response_outputs = response_outputs[0]  # (batch_size, response_len, emb_size)

        # persona
        _, max_persona_num, max_persona_len = p_input_ids.shape
        p_input_ids = p_input_ids.view(-1, max_persona_len)
        p_attention_mask = p_attention_mask.view(-1, max_persona_len)
        p_token_type_ids = p_token_type_ids.view(-1, max_persona_len)
        persona_outputs = self.bert(p_input_ids, attention_mask=p_attention_mask, token_type_ids=p_token_type_ids)
        persona_outputs = persona_outputs[0]  # (batch_size*max_persona_num, max_persona_len, emb_size)
        # 转维度
        emb_size = persona_outputs.shape[-1]
        persona_outputs = persona_outputs.view(-1, max_persona_num, max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num, max_persona_len)

        assert context_outputs.dim() == 3 and response_outputs.dim() == 3 and persona_outputs.dim() == 4
        assert c_attention_mask.dim() == 2 and r_attention_mask.dim() == 2 and p_attention_mask.dim() == 3

        # mask类型要转变下
        c_attention_mask = c_attention_mask.float()
        r_attention_mask = r_attention_mask.float()
        p_attention_mask = p_attention_mask.float()

        if train_mode:
            num_candidiates = context_outputs.shape[0]
            batch_size = num_candidiates
            # 先扩充 context和 personas
            context_outputs = context_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, context_len, emb_size)
            c_attention_mask = c_attention_mask.repeat_interleave(num_candidiates, dim=0)
            persona_outputs = persona_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, max_persona_num, num_persona_len, emb_size)
            p_attention_mask = p_attention_mask.repeat_interleave(num_candidiates, dim=0)

            # 看清楚response扩展的维度是不一样的
            _, response_len, emb_size = response_outputs.shape
            response_outputs = response_outputs.unsqueeze(0).repeat(num_candidiates, 1, 1, 1).reshape(-1, response_len,
                                                                                                      emb_size)
            r_attention_mask = r_attention_mask.unsqueeze(0).repeat(num_candidiates, 1, 1).reshape(-1, response_len)

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (2) personas 和 response交互
            # 这里response分别和每一个persona交互
            p_attend_r_list = []
            r_attend_p_list = []
            for i in range(max_persona_num):
                one_p_attend_r, one_r_attend_p = match(persona_outputs[:, i], response_outputs, p_attention_mask[:, i],
                                                       r_attention_mask)
                p_attend_r_list.append(one_p_attend_r)
                r_attend_p_list.append(one_r_attend_p)
            p_attend_r = torch.stack(p_attend_r_list,
                                     dim=1)  # (batch_size*num_candidates, max_persona_num, num_persona_len, emb_size)
            r_attend_p = torch.stack(r_attend_p_list, dim=1)
            p_attend_r = torch.mean(p_attend_r, dim=1)  # (batch_size*num_candidates, num_persona_len, emb_size)
            r_attend_p = torch.mean(r_attend_p, dim=1)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
            # 这里需要再用一次mean
            # p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            p_mean_max = torch.mean(p_attend_r, dim=1)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            # (4) 计算logits
            r_final = torch.cat([r_mean_max_c, r_mean_max_p], dim=-1)  # (batch_size*num_candidates,2*emb_size)
            c_p_final = torch.cat([c_mean_max, p_mean_max], dim=-1)  # (batch_size*num_candidates,2*emb_size)
            logits = torch.bmm(r_final.unsqueeze(1), c_p_final.unsqueeze(-1)).reshape(batch_size, num_candidiates)

            # 计算loss
            targets = torch.arange(batch_size, dtype=torch.long, device=context_outputs.device)
            loss = F.cross_entropy(logits, targets)

        else:
            # 验证模式
            # 一般设置batch_size为20，因为是从20个选一个出来，而且第一个是正样本，只拿第一个就可以了
            batch_size = context_outputs.shape[0]

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (2) personas 和 response交互
            # 这里response分别和每一个persona交互
            p_attend_r_list = []
            r_attend_p_list = []
            for i in range(max_persona_num):
                one_p_attend_r, one_r_attend_p = match(persona_outputs[:, i], response_outputs, p_attention_mask[:, i],
                                                       r_attention_mask)
                p_attend_r_list.append(one_p_attend_r)
                r_attend_p_list.append(one_r_attend_p)
            p_attend_r = torch.stack(p_attend_r_list,
                                     dim=1)  # (batch_size*num_candidates, max_persona_num, num_persona_len, emb_size)
            r_attend_p = torch.stack(r_attend_p_list, dim=1)
            p_attend_r = torch.mean(p_attend_r, dim=1)  # (batch_size*num_candidates, num_persona_len, emb_size)
            r_attend_p = torch.mean(r_attend_p, dim=1)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
            # 这里需要再用一次mean
            # p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            p_mean_max = torch.mean(p_attend_r, dim=1)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            # (4) 计算logits
            r_final = torch.cat([r_mean_max_c, r_mean_max_p], dim=-1)  # (batch_size,2*emb_size)
            c_p_final = torch.cat([c_mean_max, p_mean_max], dim=-1)  # (batch_size,2*emb_size)

            logits = torch.bmm(r_final.unsqueeze(1), c_p_final.unsqueeze(-1)).squeeze(
                2)  # (batch_size, 1 , 1) -> (batch_size, 1)
            loss = None

        return logits, loss


class BertForPersonaResponseSelection_twoloss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        # self.init_weights()

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,
            r_input_ids=None,
            r_attention_mask=None,
            r_token_type_ids=None,
            p_input_ids=None,
            p_attention_mask=None,
            p_token_type_ids=None,
            labels=None,
            train_mode=True,
            aggregate_mode="mean"  # mean / max / mean_max
    ):
        # context
        context_outputs = self.bert(c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        context_outputs = context_outputs[0]  # (batch_size, context_len, emb_size)

        # response
        response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
        response_outputs = response_outputs[0]  # (batch_size, response_len, emb_size)

        # persona
        _, max_persona_num, max_persona_len = p_input_ids.shape
        p_input_ids = p_input_ids.view(-1, max_persona_len)
        p_attention_mask = p_attention_mask.view(-1, max_persona_len)
        p_token_type_ids = p_token_type_ids.view(-1, max_persona_len)
        persona_outputs = self.bert(p_input_ids, attention_mask=p_attention_mask, token_type_ids=p_token_type_ids)
        persona_outputs = persona_outputs[0]  # (batch_size*max_persona_num, max_persona_len, emb_size)
        # 转维度
        emb_size = persona_outputs.shape[-1]
        persona_outputs = persona_outputs.view(-1, max_persona_num, max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num, max_persona_len)

        assert context_outputs.dim() == 3 and response_outputs.dim() == 3 and persona_outputs.dim() == 4
        assert c_attention_mask.dim() == 2 and r_attention_mask.dim() == 2 and p_attention_mask.dim() == 3

        # mask类型要转变下
        c_attention_mask = c_attention_mask.float()
        r_attention_mask = r_attention_mask.float()
        p_attention_mask = p_attention_mask.float()

        if train_mode:
            num_candidiates = context_outputs.shape[0]
            batch_size = num_candidiates
            # 先扩充 context和 personas
            context_outputs = context_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, context_len, emb_size)
            c_attention_mask = c_attention_mask.repeat_interleave(num_candidiates, dim=0)
            persona_outputs = persona_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, max_persona_num, num_persona_len, emb_size)
            p_attention_mask = p_attention_mask.repeat_interleave(num_candidiates, dim=0)

            # 看清楚response扩展的维度是不一样的
            _, response_len, emb_size = response_outputs.shape
            response_outputs = response_outputs.unsqueeze(0).repeat(num_candidiates, 1, 1, 1).reshape(-1, response_len,
                                                                                                      emb_size)
            r_attention_mask = r_attention_mask.unsqueeze(0).repeat(num_candidiates, 1, 1).reshape(-1, response_len)

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (2) personas 和 response交互
            # 这里response分别和每一个persona交互
            p_attend_r_list = []
            r_attend_p_list = []
            for i in range(max_persona_num):
                one_p_attend_r, one_r_attend_p = match(persona_outputs[:, i], response_outputs, p_attention_mask[:, i],
                                                       r_attention_mask)
                p_attend_r_list.append(one_p_attend_r)
                r_attend_p_list.append(one_r_attend_p)
            p_attend_r = torch.stack(p_attend_r_list,
                                     dim=1)  # (batch_size*num_candidates, max_persona_num, num_persona_len, emb_size)
            r_attend_p = torch.stack(r_attend_p_list, dim=1)
            p_attend_r = torch.mean(p_attend_r, dim=1)  # (batch_size*num_candidates, num_persona_len, emb_size)
            r_attend_p = torch.mean(r_attend_p, dim=1)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
            # 这里需要再用一次mean
            # p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            p_mean_max = torch.mean(p_attend_r, dim=1)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            # (4) 计算logits
            logtits_cr = torch.bmm(c_mean_max.unsqueeze(1), r_mean_max_c.unsqueeze(-1)).reshape(batch_size,
                                                                                                num_candidiates)
            logtits_pr = torch.bmm(p_mean_max.unsqueeze(1), r_mean_max_p.unsqueeze(-1)).reshape(batch_size,
                                                                                                num_candidiates)
            logits = logtits_cr + logtits_pr

            # 计算loss
            targets = torch.arange(batch_size, dtype=torch.long, device=context_outputs.device)
            loss1 = F.cross_entropy(logtits_cr, targets)
            loss2 = F.cross_entropy(logtits_pr, targets)
            loss = loss1 + loss2

        else:
            # 验证模式
            # 一般设置batch_size为20，因为是从20个选一个出来，而且第一个是正样本，只拿第一个就可以了
            batch_size = context_outputs.shape[0]

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (2) personas 和 response交互
            # 这里response分别和每一个persona交互
            p_attend_r_list = []
            r_attend_p_list = []
            for i in range(max_persona_num):
                one_p_attend_r, one_r_attend_p = match(persona_outputs[:, i], response_outputs, p_attention_mask[:, i],
                                                       r_attention_mask)
                p_attend_r_list.append(one_p_attend_r)
                r_attend_p_list.append(one_r_attend_p)
            p_attend_r = torch.stack(p_attend_r_list,
                                     dim=1)  # (batch_size*num_candidates, max_persona_num, num_persona_len, emb_size)
            r_attend_p = torch.stack(r_attend_p_list, dim=1)
            p_attend_r = torch.mean(p_attend_r, dim=1)  # (batch_size*num_candidates, num_persona_len, emb_size)
            r_attend_p = torch.mean(r_attend_p, dim=1)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
            # 这里需要再用一次mean
            # p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            p_mean_max = torch.mean(p_attend_r, dim=1)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            # (4) 计算logits
            logtits_cr = torch.bmm(c_mean_max.unsqueeze(1), r_mean_max_c.unsqueeze(-1)).squeeze(
                2)  # (batch_size, 1 , 1) -> (batch_size, 1)
            logtits_pr = torch.bmm(p_mean_max.unsqueeze(1), r_mean_max_p.unsqueeze(-1)).squeeze(2)
            logits = logtits_cr + logtits_pr
            loss = None

        return logits, loss


from transformers import BertForSequenceClassification

class BertForPersonaResponseSelection_multi_task(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.classifier1 = nn.Linear(config.hidden_size * 2, 256)
        self.dropout = nn.Dropout(0.2)
        self.classifier2 = nn.Linear(256, 1)
        self.init_weights()

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,

            match_p_input_ids=None,
            match_p_attention_mask=None,
            match_p_token_type_ids=None,
            match_true_index=None,

            pos_p_input_ids=None,
            pos_p_attention_mask=None,
            pos_p_token_type_ids=None,

            neg_p_input_ids=None,
            neg_p_attention_mask=None,
            neg_p_token_type_ids=None,

            aggregate_mode="mean",  # mean / max / mean_max
    ):
        # context
        conversation_outputs = self.bert(c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        conversation_outputs = conversation_outputs[0]  # (batch_size, context_len, emb_size)

        # 先处理pos_neg任务
        # pos_personas
        assert pos_p_input_ids.shape[1] == neg_p_input_ids.shape[1]
        _, max_persona_num, max_persona_len = pos_p_input_ids.shape
        pos_p_input_ids = pos_p_input_ids.view(-1, max_persona_len)
        pos_p_attention_mask = pos_p_attention_mask.view(-1, max_persona_len)
        pos_p_token_type_ids = pos_p_token_type_ids.view(-1, max_persona_len)
        pos_persona_outputs = self.bert(pos_p_input_ids, attention_mask=pos_p_attention_mask,
                                        token_type_ids=pos_p_token_type_ids)
        pos_persona_outputs = pos_persona_outputs[0]  # (batch_size*max_persona_num, max_persona_len, emb_size)
        # 转维度
        emb_size = pos_persona_outputs.shape[-1]
        pos_persona_outputs = pos_persona_outputs.view(-1, max_persona_num, max_persona_len, emb_size)
        pos_p_attention_mask = pos_p_attention_mask.view(-1, max_persona_num, max_persona_len)

        # neg_personas
        neg_p_input_ids = neg_p_input_ids.view(-1, max_persona_len)
        neg_p_attention_mask = neg_p_attention_mask.view(-1, max_persona_len)
        neg_p_token_type_ids = neg_p_token_type_ids.view(-1, max_persona_len)
        neg_persona_outputs = self.bert(neg_p_input_ids, attention_mask=neg_p_attention_mask,
                                        token_type_ids=neg_p_token_type_ids)
        neg_persona_outputs = neg_persona_outputs[0]  # (batch_size*max_persona_num, max_persona_len, emb_size)
        # 转维度
        emb_size = neg_persona_outputs.shape[-1]
        neg_persona_outputs = neg_persona_outputs.view(-1, max_persona_num, max_persona_len, emb_size)
        neg_p_attention_mask = neg_p_attention_mask.view(-1, max_persona_num, max_persona_len)

        # mask类型要转变下
        c_attention_mask = c_attention_mask.float()
        pos_p_attention_mask = pos_p_attention_mask.float()
        neg_p_attention_mask = neg_p_attention_mask.float()

        # ================================================================================
        # (1) pos_personas 和 conversation 交互
        # 这里conversation分别和每一个pos_persona交互
        pos_p_attend_c_list = []
        c_attend_pos_p_list = []
        for i in range(max_persona_num):
            one_p_attend_c, one_c_attend_p = match(pos_persona_outputs[:, i], conversation_outputs,
                                                   pos_p_attention_mask[:, i],
                                                   c_attention_mask)
            pos_p_attend_c_list.append(one_p_attend_c)
            c_attend_pos_p_list.append(one_c_attend_p)
        pos_p_attend_c = torch.stack(pos_p_attend_c_list,
                                     dim=1)  # (batch_size, max_persona_num, num_persona_len, emb_size)
        c_attend_pos_p = torch.stack(c_attend_pos_p_list, dim=1)
        pos_p_attend_c = torch.mean(pos_p_attend_c, dim=1)  # (batch_size, num_persona_len, emb_size)
        c_attend_pos_p = torch.mean(c_attend_pos_p, dim=1)

        # (2) aggregate 为了减少维度 先用mean
        pos_p_mean = torch.mean(pos_p_attend_c, dim=1)  # (batch_size, emb_size)
        c_mean_pos_p = aggregate(aggregate_mode, c_attend_pos_p, c_attention_mask)

        pos_p_r_logits = torch.cat([pos_p_mean, c_mean_pos_p], dim=1)
        pos_p_r_logits = self.dropout(pos_p_r_logits)
        pos_p_r_logits = self.classifier1(pos_p_r_logits)
        pos_p_r_logits = torch.relu(pos_p_r_logits)

        pos_p_r_logits = self.dropout(pos_p_r_logits)
        pos_p_r_logits = self.classifier2(pos_p_r_logits)  # (batch_size, 1)
        pos_p_r_logits = torch.sigmoid(pos_p_r_logits)

        # ================================================================================
        # (3) neg_personas 和 conversation 交互
        # 这里conversation分别和每一个neg_persona交互
        neg_p_attend_c_list = []
        c_attend_neg_p_list = []
        for i in range(max_persona_num):
            one_p_attend_c, one_c_attend_p = match(neg_persona_outputs[:, i], conversation_outputs,
                                                   neg_p_attention_mask[:, i],
                                                   c_attention_mask)
            neg_p_attend_c_list.append(one_p_attend_c)
            c_attend_neg_p_list.append(one_c_attend_p)
        neg_p_attend_c = torch.stack(neg_p_attend_c_list,
                                     dim=1)  # (batch_size, max_persona_num, num_persona_len, emb_size)
        c_attend_neg_p = torch.stack(c_attend_neg_p_list, dim=1)
        neg_p_attend_c = torch.mean(neg_p_attend_c, dim=1)  # (batch_size, num_persona_len, emb_size)
        c_attend_neg_p = torch.mean(c_attend_neg_p, dim=1)

        # (4) aggregate 为了减少维度 先用mean
        neg_p_mean = torch.mean(neg_p_attend_c, dim=1)  # (batch_size, emb_size)
        c_mean_neg_p = aggregate(aggregate_mode, c_attend_neg_p, c_attention_mask)

        neg_p_r_logits = torch.cat([neg_p_mean, c_mean_neg_p], dim=1)
        neg_p_r_logits = self.dropout(neg_p_r_logits)
        neg_p_r_logits = self.classifier1(neg_p_r_logits)
        neg_p_r_logits = torch.relu(neg_p_r_logits)

        neg_p_r_logits = self.dropout(neg_p_r_logits)
        neg_p_r_logits = self.classifier2(neg_p_r_logits)  # (batch_size, 1)
        neg_p_r_logits = torch.sigmoid(neg_p_r_logits)

        pos_neg_loss = 0.4 - pos_p_r_logits + neg_p_r_logits
        pos_neg_loss = pos_neg_loss.squeeze()
        zero_tensors = torch.zeros(pos_neg_loss.shape[0], dtype=torch.float, device=conversation_outputs.device)
        pos_neg_loss = torch.where(pos_neg_loss > 0, pos_neg_loss, zero_tensors)
        pos_neg_loss = torch.mean(pos_neg_loss)

        # ======================================================================
        # (5) match_personas 和 conversation交互
        # match_personas
        _, match_persona_num, max_persona_len = match_p_input_ids.shape
        match_p_input_ids = match_p_input_ids.view(-1, max_persona_len)
        match_p_attention_mask = match_p_attention_mask.view(-1, max_persona_len)
        match_p_token_type_ids = match_p_token_type_ids.view(-1, max_persona_len)
        match_persona_outputs = self.bert(match_p_input_ids, attention_mask=match_p_attention_mask,
                                          token_type_ids=match_p_token_type_ids)
        match_persona_outputs = match_persona_outputs[0]  # (batch_size*match_persona_num, max_persona_len, emb_size)
        # 转维度
        emb_size = match_persona_outputs.shape[-1]
        match_persona_outputs = match_persona_outputs.view(-1, match_persona_num, max_persona_len, emb_size)
        match_p_attention_mask = match_p_attention_mask.view(-1, match_persona_num, max_persona_len)
        match_p_attention_mask = match_p_attention_mask.float()

        match_logits_list = []
        for i in range(match_persona_num):
            one_p_attend_c, one_c_attend_p = match(match_persona_outputs[:, i], conversation_outputs,
                                                   match_p_attention_mask[:, i],
                                                   c_attention_mask)
            p_mean = aggregate(aggregate_mode, one_p_attend_c, match_p_attention_mask[:, i])
            c_mean_match_p = aggregate(aggregate_mode, one_c_attend_p, c_attention_mask)

            one_logit = torch.matmul(p_mean.unsqueeze(1), c_mean_match_p.unsqueeze(-1)).squeeze(2)  # (batch_size,1)
            match_logits_list.append(one_logit)

        match_logits = torch.cat(match_logits_list, dim=1)  # (batch_size, match_persona_num)
        match_loss = F.cross_entropy(match_logits, match_true_index.view(-1))

        total_loss = pos_neg_loss + match_loss

        return total_loss, pos_neg_loss, match_loss


class BertForPersonaResponseSelection_DIM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.classifier2 = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,
            r_input_ids=None,
            r_attention_mask=None,
            r_token_type_ids=None,
            p_input_ids=None,
            p_attention_mask=None,
            p_token_type_ids=None,
            labels=None,
            train_mode=True,
            aggregate_mode="mean"  # mean / max / mean_max
    ):
        # context
        context_outputs = self.bert(c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        context_outputs = context_outputs[0]  # (batch_size, context_len, emb_size)

        # response
        response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
        response_outputs = response_outputs[0]  # (batch_size, response_len, emb_size)

        # persona
        _, max_persona_num, max_persona_len = p_input_ids.shape
        p_input_ids = p_input_ids.view(-1, max_persona_len)
        p_attention_mask = p_attention_mask.view(-1, max_persona_len)
        p_token_type_ids = p_token_type_ids.view(-1, max_persona_len)
        persona_outputs = self.bert(p_input_ids, attention_mask=p_attention_mask, token_type_ids=p_token_type_ids)
        persona_outputs = persona_outputs[0]  # (batch_size*max_persona_num, max_persona_len, emb_size)
        # 转维度
        emb_size = persona_outputs.shape[-1]
        persona_outputs = persona_outputs.view(-1, max_persona_num, max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num, max_persona_len)
        # persona 拼接起来
        persona_outputs = persona_outputs.view(-1, max_persona_num * max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num * max_persona_len)

        assert context_outputs.dim() == 3 and response_outputs.dim() == 3 and persona_outputs.dim() == 3
        assert c_attention_mask.dim() == 2 and r_attention_mask.dim() == 2 and p_attention_mask.dim() == 2

        # mask类型要转变下
        c_attention_mask = c_attention_mask.float()
        r_attention_mask = r_attention_mask.float()
        p_attention_mask = p_attention_mask.float()

        if train_mode:
            num_candidiates = context_outputs.shape[0]
            batch_size = num_candidiates
            # 先扩充 context和 personas
            context_outputs = context_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, context_len, emb_size)
            c_attention_mask = c_attention_mask.repeat_interleave(num_candidiates, dim=0)
            persona_outputs = persona_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, max_persona_num*num_persona_len, emb_size)
            p_attention_mask = p_attention_mask.repeat_interleave(num_candidiates, dim=0)

            # 看清楚response扩展的维度是不一样的
            _, response_len, emb_size = response_outputs.shape
            response_outputs = response_outputs.unsqueeze(0).repeat(num_candidiates, 1, 1, 1).reshape(-1, response_len,
                                                                                                      emb_size)
            r_attention_mask = r_attention_mask.unsqueeze(0).repeat(num_candidiates, 1, 1).reshape(-1, response_len)

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (2) personas 和 response交互
            p_attend_r, r_attend_p = match(persona_outputs, response_outputs, p_attention_mask, r_attention_mask)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
            p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            join_features = torch.cat([r_mean_max_c, c_mean_max, r_mean_max_p, p_mean_max],
                                      dim=1)  # (batch_size*num_candidates,4*emb_size)
            join_features = self.dropout(join_features)
            join_features = self.classifier1(join_features)
            join_features = torch.relu(join_features)
            join_features = self.dropout(join_features)
            logits = self.classifier2(join_features)  # (batch_size*num_candidates,1)
            logits = logits.reshape(batch_size, num_candidiates)

            # 计算loss
            targets = torch.arange(batch_size, dtype=torch.long, device=context_outputs.device)
            loss = F.cross_entropy(logits, targets)

        else:
            # 验证模式
            # 一般设置batch_size为20，因为是从20个选一个出来，而且第一个是正样本，只拿第一个就可以了
            batch_size = context_outputs.shape[0]

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (2) personas 和 response交互
            p_attend_r, r_attend_p = match(persona_outputs, response_outputs, p_attention_mask, r_attention_mask)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
            p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            join_features = torch.cat([r_mean_max_c, c_mean_max, r_mean_max_p, p_mean_max],
                                      dim=1)  # (batch_size, 4*emb_size)
            join_features = self.dropout(join_features)
            join_features = self.classifier1(join_features)
            join_features = torch.relu(join_features)
            join_features = self.dropout(join_features)
            logits = self.classifier2(join_features)  # (batch_size,1)
            loss = None

        return logits, loss


class BertForPersonaResponseSelection_twologits(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.classifier = nn.Linear(2, 1)
        self.init_weights()

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,
            r_input_ids=None,
            r_attention_mask=None,
            r_token_type_ids=None,
            p_input_ids=None,
            p_attention_mask=None,
            p_token_type_ids=None,
            labels=None,
            train_mode=True,
            aggregate_mode="mean"  # mean / max / mean_max
    ):
        # context
        context_outputs = self.bert(c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        context_outputs = context_outputs[0]  # (batch_size, context_len, emb_size)

        # response
        response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
        response_outputs = response_outputs[0]  # (batch_size, response_len, emb_size)
        cls_response_outputs = response_outputs[:, 0]  # (batch_size, emb_size)

        # persona
        _, max_persona_num, max_persona_len = p_input_ids.shape
        p_input_ids = p_input_ids.view(-1, max_persona_len)
        p_attention_mask = p_attention_mask.view(-1, max_persona_len)
        p_token_type_ids = p_token_type_ids.view(-1, max_persona_len)
        persona_outputs = self.bert(p_input_ids, attention_mask=p_attention_mask, token_type_ids=p_token_type_ids)
        persona_outputs = persona_outputs[0]  # (batch_size*max_persona_num, max_persona_len, emb_size)
        cls_persona_outputs = persona_outputs[:, 0]  # (batch_size*max_persona_num, emb_size)
        # 转维度
        emb_size = persona_outputs.shape[-1]
        cls_persona_outputs = cls_persona_outputs.view(-1, max_persona_num, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num, max_persona_len)
        # persona mask 只需要统计下有多少个persona
        p_num_mask = p_attention_mask.sum(dim=-1)
        p_num_mask = (p_num_mask > 0).float()  # (batch_size, max_persona_num)
        c_attention_mask = c_attention_mask.float()

        assert context_outputs.dim() == 3 and cls_response_outputs.dim() == 2 and cls_persona_outputs.dim() == 3
        assert c_attention_mask.dim() == 2 and p_num_mask.dim() == 2

        if train_mode:
            num_candidiates = context_outputs.shape[0]
            batch_size = num_candidiates
            # 先扩充 context和 personas
            context_outputs = context_outputs.repeat_interleave(num_candidiates,
                                                                dim=0)  # (batch_size*num_candidates, context_len, emb_size)
            c_attention_mask = c_attention_mask.repeat_interleave(num_candidiates, dim=0)
            cls_persona_outputs = cls_persona_outputs.repeat_interleave(num_candidiates,
                                                                        dim=0)  # (batch_size*num_candidates, max_persona_num, emb_size)
            p_num_mask = p_num_mask.repeat_interleave(num_candidiates,
                                                      dim=0)  # (batch_size*num_candidates, max_persona_num)

            # 看清楚response扩展的维度是不一样的
            _, emb_size = cls_response_outputs.shape
            cls_response_outputs = cls_response_outputs.unsqueeze(0).repeat(num_candidiates, 1, 1).reshape(-1, emb_size)

            # (1) context聚合
            attn = torch.matmul(cls_response_outputs.unsqueeze(1),
                                context_outputs.transpose(1, 2))  # (batch_size*num_candidates, 1, context_len)
            attn_mask = c_attention_mask.unsqueeze(1)
            agg_context_outputs = torch.softmax(attn * attn_mask + (-1e9) * (1 - attn_mask), dim=2)
            agg_context_outputs = torch.matmul(agg_context_outputs,
                                               context_outputs)  # (batch_size*num_candidates, 1, emb_size)

            # (2) personas聚合
            attn = torch.matmul(cls_response_outputs.unsqueeze(1), cls_persona_outputs.transpose(1, 2))
            attn_mask = p_num_mask.unsqueeze(1)
            agg_personas_outputs = torch.softmax(attn * attn_mask * (-1e9) * (1 - attn_mask), dim=2)
            agg_personas_outputs = torch.matmul(agg_personas_outputs,
                                                cls_persona_outputs)  # (batch_size*num_candidates, 1, emb_size)

            # (3)
            c_r_logits = torch.matmul(agg_context_outputs,
                                      cls_response_outputs.unsqueeze(2))  # (batch_size*num_candidates, 1, 1)
            p_r_logits = torch.matmul(agg_personas_outputs,
                                      cls_response_outputs.unsqueeze(2))  # (batch_size*num_candidates, 1, 1)
            c_r_logits = c_r_logits.squeeze(2)
            p_r_logits = p_r_logits.squeeze(2)

            logits = torch.cat([c_r_logits, p_r_logits], dim=1)
            logits = self.classifier(logits)
            logits = logits.reshape(batch_size, num_candidiates)

            # 计算loss
            targets = torch.arange(batch_size, dtype=torch.long, device=context_outputs.device)
            loss = F.cross_entropy(logits, targets)

        else:
            # 验证模式
            # 一般设置batch_size为20，因为是从20个选一个出来，而且第一个是正样本，只拿第一个就可以了
            batch_size = context_outputs.shape[0]

            # (1) context聚合
            attn = torch.matmul(cls_response_outputs.unsqueeze(1),
                                context_outputs.transpose(1, 2))  # (batch_size, 1, context_len)
            attn_mask = c_attention_mask.unsqueeze(1)
            agg_context_outputs = torch.softmax(attn * attn_mask + (-1e9) * (1 - attn_mask), dim=2)
            agg_context_outputs = torch.matmul(agg_context_outputs,
                                               context_outputs)  # (batch_size, 1, emb_size)

            # (2) personas聚合
            attn = torch.matmul(cls_response_outputs.unsqueeze(1), cls_persona_outputs.transpose(1, 2))
            attn_mask = p_num_mask.unsqueeze(1)
            agg_personas_outputs = torch.softmax(attn * attn_mask * (-1e9) * (1 - attn_mask), dim=2)
            agg_personas_outputs = torch.matmul(agg_personas_outputs,
                                                cls_persona_outputs)  # (batch_size, 1, emb_size)

            # (3)
            c_r_logits = torch.matmul(agg_context_outputs,
                                      cls_response_outputs.unsqueeze(2))  # (batch_size, 1, 1)
            p_r_logits = torch.matmul(agg_personas_outputs,
                                      cls_response_outputs.unsqueeze(2))  # (batch_size, 1, 1)
            c_r_logits = c_r_logits.squeeze(2)
            p_r_logits = p_r_logits.squeeze(2)

            logits = torch.cat([c_r_logits, p_r_logits], dim=1)
            logits = self.classifier(logits)
            loss = None

        return logits, loss


class BertForPersonaResponseSelection_lessmemory(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        # self.init_weights()

    def forward(
            self,
            c_input_ids=None,
            c_attention_mask=None,
            c_token_type_ids=None,
            r_input_ids=None,
            r_attention_mask=None,
            r_token_type_ids=None,
            p_input_ids=None,
            p_attention_mask=None,
            p_token_type_ids=None,
            labels=None,
            train_mode=True,
            aggregate_mode="mean"  # mean / max / mean_max
    ):
        # context
        context_outputs = self.bert(c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        context_outputs = context_outputs[0]  # (batch_size, context_len, emb_size)

        # response
        response_outputs = self.bert(r_input_ids, attention_mask=r_attention_mask, token_type_ids=r_token_type_ids)
        response_outputs = response_outputs[0]  # (batch_size, response_len, emb_size)

        # persona
        _, max_persona_num, max_persona_len = p_input_ids.shape
        p_input_ids = p_input_ids.view(-1, max_persona_len)
        p_attention_mask = p_attention_mask.view(-1, max_persona_len)
        p_token_type_ids = p_token_type_ids.view(-1, max_persona_len)
        persona_outputs = self.bert(p_input_ids, attention_mask=p_attention_mask, token_type_ids=p_token_type_ids)
        persona_outputs = persona_outputs[0]  # (batch_size*max_persona_num, max_persona_len, emb_size)
        # 转维度
        emb_size = persona_outputs.shape[-1]
        persona_outputs = persona_outputs.view(-1, max_persona_num, max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num, max_persona_len)
        # persona 拼接起来
        persona_outputs = persona_outputs.view(-1, max_persona_num * max_persona_len, emb_size)
        p_attention_mask = p_attention_mask.view(-1, max_persona_num * max_persona_len)

        assert context_outputs.dim() == 3 and response_outputs.dim() == 3 and persona_outputs.dim() == 3
        assert c_attention_mask.dim() == 2 and r_attention_mask.dim() == 2 and p_attention_mask.dim() == 2

        # mask类型要转变下
        c_attention_mask = c_attention_mask.float()
        r_attention_mask = r_attention_mask.float()
        p_attention_mask = p_attention_mask.float()

        if train_mode:
            num_candidiates = context_outputs.shape[0]
            batch_size = num_candidiates
            logits_list = []
            for i in range(num_candidiates):
                tmp_context_outputs = context_outputs[i]
                tmp_context_outputs = tmp_context_outputs.unsqueeze(0).repeat(num_candidiates, 1, 1)
                tmp_c_attention_mask = c_attention_mask[i]
                tmp_c_attention_mask = tmp_c_attention_mask.unsqueeze(0).repeat(num_candidiates, 1)

                tmp_persona_outputs = persona_outputs[i]
                tmp_persona_outputs = tmp_persona_outputs.unsqueeze(0).repeat(num_candidiates, 1, 1)
                tmp_p_attention_mask = p_attention_mask[i]
                tmp_p_attention_mask = tmp_p_attention_mask.unsqueeze(0).repeat(num_candidiates, 1)

                # (1) context 和 response交互
                c_attend_r, r_attend_c = match(tmp_context_outputs, response_outputs, tmp_c_attention_mask,
                                               r_attention_mask)

                # (2) personas 和 response交互
                p_attend_r, r_attend_p = match(tmp_persona_outputs, response_outputs, tmp_p_attention_mask,
                                               r_attention_mask)

                # (3) aggregate 为了减少维度 先用mean
                c_mean_max = aggregate(aggregate_mode, c_attend_r, tmp_c_attention_mask)
                r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
                p_mean_max = aggregate(aggregate_mode, p_attend_r, tmp_p_attention_mask)
                r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

                r_final = torch.cat([r_mean_max_c, r_mean_max_p], dim=-1)  # (num_candidates,2*emb_size)
                c_p_final = torch.cat([c_mean_max, p_mean_max], dim=-1)  # (num_candidates,2*emb_size)
                one_logit = torch.bmm(r_final.unsqueeze(1), c_p_final.unsqueeze(-1)).reshape(-1)
                logits_list.append(one_logit)
                print(logits_list)

            logits = torch.stack(logits_list, dim=0)
            print(logits.shape)
            print(logits)
            # 计算loss
            targets = torch.arange(batch_size, dtype=torch.long, device=context_outputs.device)
            loss = F.cross_entropy(logits, targets)

        else:
            # 验证模式
            # 一般设置batch_size为20，因为是从20个选一个出来，而且第一个是正样本，只拿第一个就可以了
            batch_size = context_outputs.shape[0]

            # (1) context 和 response交互
            c_attend_r, r_attend_c = match(context_outputs, response_outputs, c_attention_mask, r_attention_mask)

            # (2) personas 和 response交互
            p_attend_r, r_attend_p = match(persona_outputs, response_outputs, p_attention_mask, r_attention_mask)

            # (3) aggregate 为了减少维度 先用mean
            c_mean_max = aggregate(aggregate_mode, c_attend_r, c_attention_mask)
            r_mean_max_c = aggregate(aggregate_mode, r_attend_c, r_attention_mask)
            p_mean_max = aggregate(aggregate_mode, p_attend_r, p_attention_mask)
            r_mean_max_p = aggregate(aggregate_mode, r_attend_p, r_attention_mask)

            r_final = torch.cat([r_mean_max_c, r_mean_max_p], dim=-1)  # (batch_size,2*emb_size)
            c_p_final = torch.cat([c_mean_max, p_mean_max], dim=-1)  # (batch_size,2*emb_size)

            logits = torch.bmm(r_final.unsqueeze(1), c_p_final.unsqueeze(-1)).squeeze(
                2)  # (batch_size, 1 , 1) -> (batch_size, 1)
            loss = None

        return logits, loss
