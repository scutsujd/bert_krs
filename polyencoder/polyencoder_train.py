import os
import argparse
import collections
import logging
import datetime
import random
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, WarmupLinearSchedule
from LM_paper.Persona_Response_Selection_noneg.polyencoder.Polyencoder import Polyencoder_v1, Biencoder_v1
from LM_paper.Persona_Response_Selection_noneg.polyencoder.polyencoder_utils import InputExample, \
    load_and_cache_examples
from LM_paper.Persona_Response_Selection_noneg.polyencoder.polyencoder_utils import mean_average_precision, \
    mean_reciprocal_rank, \
    top_1_precision

# 暂时只有Bert
MODEL_CLASSES = {
    'bert': (BertConfig, Polyencoder_v1, BertTokenizer)}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
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
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=t_total)
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
    print("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
          args.train_batch_size * args.gradient_accumulation_steps)
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    print("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_top1_precision = 0.78
    for epoch_index in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            model.train()
            # 这里是为了防止程序错误
            if batch[0].shape[0] != args.train_batch_size:
                print("batch[0].shape[0] != args.train_batch_size")
                print(epoch_index, step)
                print(batch[0].shape[0])
                continue
            if epoch_index == 0 and step == 0:
                for t in batch:
                    print(t)
                    print("=" * 50)
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'c_input_ids': batch[0],
                      'c_attention_mask': batch[1],
                      'c_token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM, DistilBERT and RoBERTa don't use segment_ids

                      'r_input_ids': batch[3],
                      'r_attention_mask': batch[4],
                      'r_token_type_ids': batch[5] if args.model_type in ['bert', 'xlnet'] else None,

                      'labels': batch[6],
                      'train_mode': True
                      }

            logits, loss = model(**inputs)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, lr {}, loss {}".format(time_str, global_step, scheduler.get_lr()[0],
                                                               (tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

                # 500步的时候用来测试下新的验证数据格式是否可以
                if global_step == 100 or \
                        (epoch_index >= (args.num_train_epochs - 4) and global_step % args.valid_steps == 0) or \
                        (global_step == t_total // args.num_train_epochs * 4) \
                        or (global_step == t_total // args.num_train_epochs * 6):
                    map, mrr, top1_precision, top2_precision, top5_precision = evaluate(args, model, tokenizer,
                                                                                        test=False)
                    print(
                        "valid current map: {}, mrr: {}, top1_precision: {}, top2_precision: {}, top5_precision: {}".format(
                            map, mrr, top1_precision, top2_precision, top5_precision))
                    print("=" * 100)

                    # Save model checkpoint
                    if top1_precision > best_top1_precision:
                        best_top1_precision = top1_precision
                        print("new top1 precision: ", best_top1_precision)

                        # checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        # if not os.path.exists(checkpoint_dir):
                        #     os.makedirs(checkpoint_dir)
                        # model_to_save = model.module if hasattr(model,
                        #                                         'module') else model  # Take care of distributed/parallel training
                        # model_to_save.save_pretrained(checkpoint_dir)
                        # tokenizer.save_pretrained(checkpoint_dir)
                        # print("Saving model checkpoint to %s", checkpoint_dir)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, test=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = args.task_name
    eval_output_dir = args.output_dir

    if test:
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=False, test=True)
    else:
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, test=False)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    new_out_label_ids = None
    '''
        for first_index, batch in enumerate(eval_dataloader):
        if first_index % 1000 == 0:
            print(first_index)
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'c_input_ids': batch[0],
                      'c_attention_mask': batch[1],
                      'c_token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM, DistilBERT and RoBERTa don't use segment_ids

                      'r_input_ids': batch[3],
                      'r_attention_mask': batch[4],
                      'r_token_type_ids': batch[5] if args.model_type in ['bert', 'xlnet'] else None,

                      'p_input_ids': batch[6],
                      'p_attention_mask': batch[7],
                      'p_token_type_ids': batch[8] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': batch[9],
                      'train_mode': False
                      }

            logits, loss = model(**inputs)

        if preds is None:
            assert logits.dim() == 2
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            assert logits.dim() == 2
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        if first_index < 2:
            print(logits)

    print("preds_list shape: ", preds.shape)
    print("out_label_ids shape: ", out_label_ids.shape)
    map = mean_average_precision(preds_list=preds, grouth_true_list=out_label_ids)
    mrr = mean_reciprocal_rank(preds_list=preds, grouth_true_list=out_label_ids)
    top1_precision, top2_precision, top5_precision = top_1_precision(preds_list=preds, grouth_true_list=out_label_ids)
    print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}'.format(
        map, mrr, top1_precision))
    '''
    for first_index, batch in enumerate(eval_dataloader):
        if first_index % 500 == 0:
            print(first_index)
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'c_input_ids': batch[0],
                      'c_attention_mask': batch[1],
                      'c_token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM, DistilBERT and RoBERTa don't use segment_ids

                      'r_input_ids': batch[3],
                      'r_attention_mask': batch[4],
                      'r_token_type_ids': batch[5] if args.model_type in ['bert', 'xlnet'] else None,

                      'labels': batch[6],
                      "train_mode": False
                      }

            logits, loss = model(**inputs)

        if preds is None:
            assert logits.dim() == 2
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
            # print("evaluate preds: ", preds)
            # print("evaluate out_label_ids: ", out_label_ids)

            preds = preds.reshape(-1, 1)
            # print("evaluate preds: ", preds)
            new_out_label_ids = []
            for index in out_label_ids:
                one_li = [0] * 20
                one_li[int(index)] = 1
                new_out_label_ids.extend(one_li)
            new_out_label_ids = np.array(new_out_label_ids, dtype=np.int)
            # print("evaluate new_out_label_ids: ", new_out_label_ids)
            assert preds.shape[0] == new_out_label_ids.shape[0]
        else:
            assert logits.dim() == 2
            preds = np.append(preds, logits.detach().cpu().numpy().reshape(-1, 1), axis=0)

            out_label_ids = inputs['labels'].detach().cpu().numpy()
            new_oli = []
            for index in out_label_ids:
                one_li = [0] * 20
                one_li[int(index)] = 1
                new_oli.extend(one_li)
            new_oli = np.array(new_oli, dtype=np.int)
            new_out_label_ids = np.append(new_out_label_ids, new_oli, axis=0)
            assert preds.shape[0] == new_out_label_ids.shape[0]

        # if first_index == 0:
        #     print(logits.detach().cpu().numpy()[:5])

    print("preds_list shape: ", preds.shape)
    print("new_out_label_ids shape: ", new_out_label_ids.shape)
    map = mean_average_precision(preds_list=preds, grouth_true_list=new_out_label_ids)
    mrr = mean_reciprocal_rank(preds_list=preds, grouth_true_list=new_out_label_ids)
    top1_precision, top2_precision, top5_precision = top_1_precision(preds_list=preds,
                                                                     grouth_true_list=new_out_label_ids)
    print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}'.format(
        map, mrr, top1_precision))

    return map, mrr, top1_precision, top2_precision, top5_precision


class Args():
    def __init__(self):
        self.task_name = "cmudog"
        if self.task_name == "personachat_self_original":
            self.train_data_dir = "/data/mgw_data/FIRE_data/personachat_data/personachat_processed/processed_train_self_original.txt"
            self.valid_data_dir = "/data/mgw_data/FIRE_data/personachat_data/personachat_processed/processed_valid_self_original.txt"
            self.test_data_dir = "/data/mgw_data/FIRE_data/personachat_data/personachat_processed/processed_test_self_original.txt"
        elif self.task_name == "personachat_self_revised":
            self.train_data_dir = "/data/mgw_data/FIRE_data/personachat_data/personachat_processed/processed_train_self_revised.txt"
            self.valid_data_dir = "/data/mgw_data/FIRE_data/personachat_data/personachat_processed/processed_valid_self_revised.txt"
            self.test_data_dir = "/data/mgw_data/FIRE_data/personachat_data/personachat_processed/processed_test_self_revised.txt"
        elif self.task_name == "cmudog":
            self.train_data_dir = "/data/mgw_data/FIRE_data/cmudog_data/cmudog_processed/processed_train_self_original_fullSection.txt"
            self.valid_data_dir = "/data/mgw_data/FIRE_data/cmudog_data/cmudog_processed/processed_valid_self_original_fullSection.txt"
            self.test_data_dir = "/data/mgw_data/FIRE_data/cmudog_data/cmudog_processed/processed_test_self_original_fullSection.txt"
        else:
            raise NotImplementedError

        self.model_type = "bert"  # "bert"  # "roberta" # "xlnet"
        if self.model_type == "bert":
            self.model_name_or_path = "/data/mgw_data/bert_base_uncased_model/"
        elif self.model_type == "roberta":
            self.model_name_or_path = "/data/mgw_data/roberta_base_model/"
        elif self.model_type == "xlnet":
            self.model_name_or_path = "/data/mgw_data/xlnet_base_model/"

        self.data_suffix = "polyencoder"
        self.output_suffix = "polyencoder" + "_lw"
        # self.output_suffix = "biencoder" + "_lw"
        self.output_dir = "/data/mgw_data/BERT_persona_response_selection/save_model/" + self.model_type + "_" + self.task_name + "_" + self.output_suffix
        self.data_dir = "/data/mgw_data/BERT_persona_response_selection/cached_data"

        if self.task_name == "cmudog":
            self.context_max_length = 11 * 30
            self.persona_max_length = 180
            self.max_persona_num = None
            self.response_max_length = 50
            self.dataset_size = 36159
        else:
            self.context_max_length = 15 * 20
            self.persona_max_length = 100
            self.max_persona_num = None
            self.response_max_length = 50
            self.dataset_size = 65719

        self.do_lower_case = True

        self.neg_num = 0
        self.train_batch_size = 14 if self.task_name == "cmudog" else 20  # 12  # 20
        self.eval_batch_size = 20
        self.gradient_accumulation_steps = 1
        self.learning_rate = 3e-5  # 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 10
        self.max_steps = -1
        # 默认两块GPU 总步数前10% warmup personachat训练数据：65719  cmudog训练数据：36159
        self.logging_steps = 250 if self.task_name == "cmudog" else 500  # 500  # 250

        self.warmup_steps = int(self.dataset_size * (self.neg_num + 1) * self.num_train_epochs) // \
                            (self.train_batch_size * self.gradient_accumulation_steps) // 10

        self.valid_steps = int(self.dataset_size * (self.neg_num + 1)) // \
                           (self.train_batch_size * self.gradient_accumulation_steps) // 10  # 每个epoch20%验证一次

        self.seed = 2022
        self.use_cuda = True
        self.n_gpu = 1
        self.do_train = True
        self.do_eval = False

        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.local_rank = -1


if __name__ == "__main__":
    # 这里反过来是为了让第二块gpu作为主gpu，第一块用久了 慢了一点
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

    args.task_name = args.task_name.lower()
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)

    # Training
    if args.do_train:
        # 先新建模型保存文件夹
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, test=False)
        # 构造数据用的
        # eval_dataseet = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True, test=False)
        # test_dataseet = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, test=True)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        print(" global_step = %s, average loss = %s", global_step, tr_loss)
    if args.do_eval:
        map, mrr, top1_precision, top2_precision, top5_precision = evaluate(args, model, tokenizer)
        print("current map: {}, mrr: {}, top1_precision: {}, top2_precision: {}, top5_precision: {}".format(
            map, mrr, top1_precision, top2_precision, top5_precision))
