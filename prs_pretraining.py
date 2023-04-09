import os
import argparse
import collections
import logging
from datetime import datetime
import torch
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForPreTraining
import transformers
# from LM_paper.UMS_Response_Selection.models.bert.modeling_bert import BertForPreTraining
import random
import numpy as np
from LM_paper.Persona_Response_Selection_noneg.prs_create_pretraining_data import TrainingFeature

MODEL_CLASSES = {
    'bert': (BertConfig, BertForPreTraining, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args.num_train_epochs)
    print("  Instantaneous batch size GPU = %d", args.train_batch_size)
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
          args.train_batch_size * args.gradient_accumulation_steps)
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    print("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch_index in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            model.train()
            if epoch_index == 0 and step == 0:
                for t in batch:
                    print(t)
                    print("=" * 50)
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "masked_lm_labels": batch[3],
                "next_sentence_label": batch[4]
            }

            outputs = model(**inputs)
            loss = outputs[0]  # 这里是两个loss加起来了(loss = masked_lm_loss + next_sentence_loss)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    time_str = datetime.now().isoformat()
                    print("{}: step {}, lr {}, loss {}".format(time_str, global_step, scheduler.get_lr()[0],
                                                               (tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

                if epoch_index >= (args.num_train_epochs - 3) and global_step % (t_total // args.num_train_epochs) == 0:
                    # Save model checkpoint
                    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    print("Saving model checkpoint to %s", checkpoint_dir)

    return global_step, tr_loss / global_step


class Args():
    def __init__(self):
        self.task_name = "cmudog"
        if self.task_name == "personachat_self_original":
            # 被注释掉的是第一版的预训练数据，10w数据，而且不加特殊符号，
            # self.pretrain_data_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/personachat_self_original_pretraining_data.pkl"
            # 这是优化后的预训练数据，数据量更大，加了两个特殊符号
            self.pretrain_data_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/personachat_self_original_pretraining_data_V2.pkl"
        elif self.task_name == "personachat_self_revised":
            self.pretrain_data_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/personachat_self_revised_pretraining_data_V2.pkl"
        elif self.task_name == "cmudog":
            self.pretrain_data_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/cmudog_pretraining_data_V2.pkl"
            # 消融实验用的 用来测试DA的
            # self.pretrain_data_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/cmudog_pretraining_data_V2noDA.pkl"
        else:
            raise NotImplementedError

        self.model_type = "bert"  # "bert"  # "roberta" # "xlnet"
        if self.model_type == "bert":
            self.model_name_or_path = "/data/mgw_data/bert_base_uncased_model/"
        elif self.model_type == "roberta":
            self.model_name_or_path = "/data/mgw_data/roberta_base_model/"
        elif self.model_type == "xlnet":
            self.model_name_or_path = "/data/mgw_data/xlnet_base_model/"

        # self.output_dir = "/data/mgw_data/BERT_persona_response_selection/save_model/pretrain_" + self.model_type + "_" + self.task_name
        # self.output_dir = "/data/mgw_data/BERT_persona_response_selection/save_model/pretrain_" + self.model_type + "_" + self.task_name + "_V2"

        # 这里做消融实验，只针对cmudog数据集
        # 最后那个noDA需要更换数据集
        # self.output_dir = "/data/mgw_data/BERT_persona_response_selection/save_model/pretrain_" + self.model_type + "_" + self.task_name + "_V2noCD"
        self.output_dir = "/data/mgw_data/BERT_persona_response_selection/save_model/pretrain_" + self.model_type + "_" + self.task_name + "_V2noMLM"
        # self.output_dir = "/data/mgw_data/BERT_persona_response_selection/save_model/pretrain_" + self.model_type + "_" + self.task_name + "_V2noDA"

        self.do_lower_case = True
        self.add_tokens = ['[EOU]', '[EOK]']

        self.train_batch_size = 12 if self.task_name == "cmudog" else 14  # personachat: 14
        self.gradient_accumulation_steps = 1
        self.learning_rate = 3e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 5
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 500 if self.task_name == "cmudog" else 250  # 500  # 250

        self.seed = 2022
        self.use_cuda = True
        self.n_gpu = 2
        self.do_train = True

        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.local_rank = -1


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"
    print("torch.cuda.device_count() {}".format(torch.cuda.device_count()))
    args = Args()
    print("\nTraining/evaluation parameters: ")
    for key in args.__dict__:
        print("{}={}".format(key, args.__dict__[key]))

    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        print("gpu count: ", args.n_gpu)
    else:
        device = torch.device("cpu")
    args.device = device
    print("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
          args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    set_seed(args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    # tokenizer只是保存的时候用到
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    # 增加两个特殊符号
    tokenizer.add_tokens(args.add_tokens)
    print(len(tokenizer))
    print(model.config.vocab_size)
    model.resize_token_embeddings(len(tokenizer))
    print(model.config.vocab_size)
    model.to(args.device)

    # Training
    if args.do_train:
        features = torch.load(args.pretrain_data_file)
        print("features len: ", len(features))
        # 这里直接构造dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in features], dtype=torch.long)
        all_next_sentence_labels = torch.tensor([f.next_sentence_labels for f in features], dtype=torch.long)

        train_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_masked_lm_labels,
                                      all_next_sentence_labels)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
