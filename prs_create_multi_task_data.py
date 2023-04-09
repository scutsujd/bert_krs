from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from collections import Counter

sys.path.append(os.getcwd())
import collections
import random
import argparse
import h5py
import numpy as np
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
from transformers import BertTokenizer
from LM_paper.Persona_Response_Selection_noneg.prs_utils import create_masked_lm_predictions
import torch


def tokenize(text):
    return WordPunctTokenizer().tokenize(text)


def create_pretrain_data_step1(train_file=None, save_conversation_file=None, save_persona_file=None):
    all_train_dial = []
    one_dial = []
    with open(train_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.split()[0] == "1":
                all_train_dial.append(one_dial)
                one_dial = []
            one_dial.append(line)
        all_train_dial.append(one_dial)
        all_train_dial.remove([])

    print("{} is composed of {} dialogues".format(train_file, len(all_train_dial)))

    all_dial = all_train_dial
    print("train  data {}".format(len(all_dial)))

    uniq_personas = []
    conversation_and_persona = []

    for one_dial in all_dial:
        persona = []
        context_history = []
        for line in one_dial:
            fields = line.strip().split("\t")

            if len(fields) == 1:
                persona.append(" ".join(tokenize(fields[0])[4:]))

            if len(fields) == 4:
                context = " ".join(tokenize(fields[0])[1:])
                response = " ".join(tokenize(fields[1]))  # fields[1]

                context_history.append(context)
                context_history.append(response)

        # 先把persona列表排序，然后把拼接在一起
        # personachat数据集要sorted，因为句子会被重用
        # cmudog就不用了，都是一样的
        persona = sorted(persona)
        persona = "|".join(persona)
        if persona not in uniq_personas:
            uniq_personas.append(persona)

        conversation_and_persona.append([" _eos_ ".join(context_history), persona, "1"])

    print("conversation_and_persona length {}".format(len(conversation_and_persona)))
    print("uniq_personas length {}".format(len(uniq_personas)))

    # ================================================================================
    # 统计下每个对话长度
    conversation_len_list = []
    for index, one_c_and_p in enumerate(conversation_and_persona):
        if index < 5:
            print(one_c_and_p)
        one_c = one_c_and_p[0].split(" _eos_ ")
        oc_len = 0
        for oc in one_c:
            oc_len += len(oc.split(" "))
        conversation_len_list.append(oc_len)

    print("conversation_len_list: ", conversation_len_list)
    print("avg conversation_len: ", sum(conversation_len_list) / len(conversation_len_list))
    print("conversation_len_list count: ", Counter(conversation_len_list))
    print("max", max(conversation_len_list))
    print("min", min(conversation_len_list))

    # 统计persona长度
    persona_len_list = []
    for index, one_p in enumerate(uniq_personas):
        one_p = one_p.split("|")
        op_len = 0
        for op in one_p:
            op_len += len(op.split(" "))
        persona_len_list.append(op_len)
    print("persona_len_list: ", persona_len_list)
    print("avg persona_len: ", sum(persona_len_list) / len(persona_len_list))
    print("persona_len_list count: ", Counter(persona_len_list))
    print("max", max(persona_len_list))
    print("min", min(persona_len_list))

    # ==============================================================================
    print("conversation_and_persona length {}".format(len(conversation_and_persona)))

    with open(save_conversation_file, "w", encoding="utf-8") as file:
        for one_c_and_p in conversation_and_persona:
            assert len(one_c_and_p) == 3
            file.write("\t".join(one_c_and_p) + "\n")

    with open(save_persona_file, "w", encoding="utf-8") as file:
        for one_persona in uniq_personas:
            file.write(one_persona + "\n")


class InputExample_Multi_Task:
    def __init__(self, conversation=None, match_personas=None, match_true_index=None,
                 pos_personas=None, neg_personas=None):
        self.conversation = conversation
        self.match_personas = match_personas
        self.match_true_index = match_true_index
        self.pos_personas = pos_personas
        self.neg_personas = neg_personas


class InputFeature_Multi_Task:
    def __init__(self,
                 conversation_input_ids=None, conversation_attention_mask=None, conversation_token_type_ids=None,
                 match_personas_input_ids=None, match_personas_attention_mask=None, match_personas_token_type_ids=None,
                 match_true_index=None,
                 pos_personas_input_ids=None, pos_personas_attention_mask=None, pos_personas_token_type_ids=None,
                 neg_personas_input_ids=None, neg_personas_attention_mask=None, neg_personas_token_type_ids=None
                 ):
        self.conversation_input_ids = conversation_input_ids
        self.conversation_attention_mask = conversation_attention_mask
        self.conversation_token_type_ids = conversation_token_type_ids

        self.match_personas_input_ids = match_personas_input_ids
        self.match_personas_attention_mask = match_personas_attention_mask
        self.match_personas_token_type_ids = match_personas_token_type_ids
        self.match_true_index = match_true_index

        self.pos_personas_input_ids = pos_personas_input_ids
        self.pos_personas_attention_mask = pos_personas_attention_mask
        self.pos_personas_token_type_ids = pos_personas_token_type_ids

        self.neg_personas_input_ids = neg_personas_input_ids
        self.neg_personas_attention_mask = neg_personas_attention_mask
        self.neg_personas_token_type_ids = neg_personas_token_type_ids


def create_pretrain_data_step2(args=None):
    # 这里读入personas文件
    personas_collection = []
    one_persona_collection = []
    with open(args.uniq_persona_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if index < 5:
                print(line)
            personas_collection.append(line)

            line = line.split("|")
            for one in line:
                if one not in one_persona_collection:
                    one_persona_collection.append(one)
    print("one_persona_collection len: ", len(one_persona_collection))
    random.shuffle(one_persona_collection)
    print("personas_collection len: ", len(personas_collection))
    print("one_persona_collection len: ", len(one_persona_collection))

    examples = []
    with open(args.whole_conversation_file, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index % 1000 == 0:
                print(index)
            line = line.strip()
            line = line.split("\t")
            if index < 5:
                print("=" * 50)
                print(line[0])
                print(line[1])
                print(line[2])
            assert len(line) == 3
            conversation = line[0]
            personas = line[1]
            assert personas in personas_collection
            conversation = conversation.split(" _eos_ ")
            personas = personas.split("|")
            pos_personas_join = "|".join(personas)  # 这里拼接起来是为后面选完整的personas

            # 抽九个负例
            neg_personas_join_list = []
            for one in personas:
                # 这里构建match_persona任务的数据
                match_personas_list = [one]
                while len(match_personas_list) < 10:  # 1+9
                    neg_one = random.choice(one_persona_collection)
                    if neg_one not in match_personas_list and neg_one not in personas:
                        match_personas_list.append(neg_one)

                random.shuffle(match_personas_list)
                match_true_index = match_personas_list.index(one)
                assert len(match_personas_list) == 10

                # 这里构建pos_neg任务的数据
                neg_personas_join = random.choice(personas_collection)
                while neg_personas_join == pos_personas_join or neg_personas_join in neg_personas_join_list:
                    neg_personas_join = random.choice(personas_collection)
                neg_personas_join_list.append(neg_personas_join)

                examples.append(InputExample_Multi_Task(
                    conversation=conversation,
                    match_personas=match_personas_list,
                    match_true_index=match_true_index,
                    pos_personas=pos_personas_join.split("|"),
                    neg_personas=neg_personas_join.split("|")
                ))

            if index < 5:
                for i in range(len(neg_personas_join_list)):
                    print("+" * 50)
                    print(examples[-i].conversation)
                    print(examples[-i].match_personas)
                    print(examples[-i].match_true_index)
                    print(examples[-i].pos_personas)
                    print(examples[-i].neg_personas)

    print("examples len: ", len(examples))

    # =================================================================================
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained)
    print("all special tokens: ", tokenizer.all_special_tokens)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d" % (ex_index))

        conversation = example.conversation
        conversation_tokens_list = []
        conversation_token_type_ids = []
        token_type = 0
        for one_c in conversation:
            t_one_c = tokenizer.tokenize(one_c) + ["[SEP]"]
            conversation_tokens_list += t_one_c
            conversation_token_type_ids += ([token_type] * len(t_one_c))
            token_type = int(1 - token_type)
        conversation_tokens_list = tokenizer.convert_tokens_to_ids(conversation_tokens_list)
        assert len(conversation_tokens_list) == len(conversation_token_type_ids)

        if len(conversation_tokens_list) > args.max_conversation_length - 1:  # -1是因为需要加CLS
            conversation_tokens_list = conversation_tokens_list[-(args.max_conversation_length - 1):]
            conversation_token_type_ids = conversation_token_type_ids[-(args.max_conversation_length - 1):]
            if conversation_tokens_list[0] == tokenizer.convert_tokens_to_ids("[SEP]"):
                conversation_tokens_list = conversation_tokens_list[1:]
                conversation_token_type_ids = conversation_token_type_ids[1:]

        conversation_tokens_list = [tokenizer.convert_tokens_to_ids("[CLS]")] + conversation_tokens_list
        conversation_token_type_ids = [0] + conversation_token_type_ids
        assert len(conversation_tokens_list) == len(conversation_token_type_ids)

        conversation_attention_mask = [1] * len(conversation_tokens_list)
        # padding操作
        if len(conversation_tokens_list) < args.max_conversation_length:
            padding_length = args.max_conversation_length - len(conversation_tokens_list)
            conversation_tokens_list = conversation_tokens_list + [0] * padding_length
            conversation_token_type_ids = conversation_token_type_ids + [0] * padding_length
            conversation_attention_mask = conversation_attention_mask + [0] * padding_length

        assert len(conversation_tokens_list) == args.max_conversation_length
        assert len(conversation_token_type_ids) == args.max_conversation_length
        assert len(conversation_attention_mask) == args.max_conversation_length

        # ===============================================================
        # 这个不用控制persona数量，因为已经确保是10个了
        match_personas = example.match_personas
        match_personas_tokens_list, \
        match_personas_attention_mask_list, \
        match_personas_token_type_ids_list = create_personas_data(args, match_personas, tokenizer,
                                                                  use_max_persona_num=False)

        pos_personas = example.pos_personas
        if len(pos_personas) > args.max_persona_num:
            pos_personas = pos_personas[-args.max_persona_num:]
        pos_personas_tokens_list, \
        pos_personas_attention_mask_list, \
        pos_personas_token_type_ids_list = create_personas_data(args, pos_personas, tokenizer,
                                                                use_max_persona_num=True)

        neg_personas = example.neg_personas
        if len(neg_personas) > args.max_persona_num:
            neg_personas = neg_personas[-args.max_persona_num:]
        neg_personas_tokens_list, \
        neg_personas_attention_mask_list, \
        neg_personas_token_type_ids_list = create_personas_data(args, neg_personas, tokenizer,
                                                                use_max_persona_num=True)

        if ex_index < 5:
            print("*** Example ***")
            print("example.conversation: %s " % (example.conversation))
            print("conversation_tokens_list: %s " % (conversation_tokens_list))
            print("context_attention_mask: %s " % (conversation_attention_mask))
            print("context_token_type_ids: %s " % (conversation_token_type_ids))

            print("example.match_personas: %s " % (example.match_personas))
            print("match_personas_tokens_list: %s " % (match_personas_tokens_list))
            print("match_personas_attention_mask_list: %s " % (match_personas_attention_mask_list))
            print("match_personas_token_type_ids_list: %s " % (match_personas_token_type_ids_list))
            print("example.match_personas_index: %s " % (example.match_true_index))

            print("example.pos_personas: %s " % (example.pos_personas))
            print("pos_personas_tokens_list: %s " % (pos_personas_tokens_list))
            print("pos_personas_attention_mask_list: %s " % (pos_personas_attention_mask_list))
            print("pos_personas_token_type_ids_list: %s " % (pos_personas_token_type_ids_list))

            print("example.neg_personas: %s " % (example.neg_personas))
            print("neg_personas_tokens_list: %s " % (neg_personas_tokens_list))
            print("neg_personas_attention_mask_list: %s " % (neg_personas_attention_mask_list))
            print("neg_personas_token_type_ids_list: %s " % (neg_personas_token_type_ids_list))

        features.append(InputFeature_Multi_Task(
            conversation_input_ids=conversation_tokens_list,
            conversation_attention_mask=conversation_attention_mask,
            conversation_token_type_ids=conversation_token_type_ids,

            match_personas_input_ids=match_personas_tokens_list,
            match_personas_attention_mask=match_personas_attention_mask_list,
            match_personas_token_type_ids=match_personas_token_type_ids_list,
            match_true_index=example.match_true_index,

            pos_personas_input_ids=pos_personas_tokens_list,
            pos_personas_attention_mask=pos_personas_attention_mask_list,
            pos_personas_token_type_ids=pos_personas_token_type_ids_list,

            neg_personas_input_ids=neg_personas_tokens_list,
            neg_personas_attention_mask=neg_personas_attention_mask_list,
            neg_personas_token_type_ids=neg_personas_token_type_ids_list
        ))

    torch.save(features, args.output_file)


def create_personas_data(args, personas, tokenizer, use_max_persona_num=True):
    personas_tokens_list = []
    personas_attention_mask_list = []
    personas_token_type_ids_list = []
    for one_p in personas:
        t_one_p = tokenizer.tokenize(one_p)
        if len(t_one_p) > args.max_persona_length - 2:
            t_one_p = t_one_p[:args.max_persona_length - 2]
        t_one_p = ["[CLS]"] + t_one_p + ["[SEP]"]
        t_one_p = tokenizer.convert_tokens_to_ids(t_one_p)

        one_atten_mask = [1] * len(t_one_p)
        if len(t_one_p) < args.max_persona_length:
            padding_length = args.max_persona_length - len(t_one_p)
            t_one_p = t_one_p + [0] * padding_length
            one_atten_mask = one_atten_mask + [0] * padding_length
        one_token_type_ids = [0] * args.max_persona_length

        assert len(t_one_p) == args.max_persona_length
        assert len(one_atten_mask) == args.max_persona_length
        assert len(one_token_type_ids) == args.max_persona_length

        personas_tokens_list.append(t_one_p)
        personas_attention_mask_list.append(one_atten_mask)
        personas_token_type_ids_list.append(one_token_type_ids)

    if use_max_persona_num and len(personas_tokens_list) < args.max_persona_num:
        for i in range(args.max_persona_num - len(personas_tokens_list)):
            personas_tokens_list.append([0] * args.max_persona_length)
            personas_attention_mask_list.append([0] * args.max_persona_length)
            personas_token_type_ids_list.append([0] * args.max_persona_length)

    assert len(personas_tokens_list[0]) == args.max_persona_length
    assert len(personas_attention_mask_list[0]) == args.max_persona_length
    assert len(personas_token_type_ids_list[0]) == args.max_persona_length

    return personas_tokens_list, personas_attention_mask_list, personas_token_type_ids_list


class Args:
    def __init__(self):
        self.input_file1 = "/data/mgw_data/FIRE_data/personachat_data/personachat/train_self_original.txt"
        # self.input_file1 = "/data/mgw_data/FIRE_data/personachat_data/personachat/train_self_revised.txt"
        # self.input_file1 = "/data/mgw_data/FIRE_data/cmudog_data/cmudog/train_self_original_fullSection.txt"

        # 分两步处理，第一步把train完整的conversation和persona拿出来,
        # 然后所有独立的persona也拿出来，用来构造负样本
        self.whole_conversation_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/multi_task_personachat_self_original_conversation.txt"
        self.uniq_persona_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/multi_task_personachat_self_original_persona.txt"
        # 第二步才是最终多任务的数据
        self.output_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/multi_task_personachat_self_original_data.pkl"

        # 这里先用经过mlm和nsp预训练过的模型
        # self.bert_pretrained = "/data/mgw_data/bert_base_uncased_model"
        self.bert_pretrained = "/data/mgw_data/BERT_persona_response_selection/save_model/pretrain_bert_personachat_self_original/checkpoint-21300/"
        self.do_lower_case = True
        self.do_whole_word_mask = True
        self.max_conversation_length = 335  # 算上[SEP]的
        self.max_persona_length = 25  # 一个persona的长度，算上[SEP]的
        self.max_persona_num = 5

        self.random_seed = 2022


if __name__ == "__main__":
    args = Args()
    random.seed(args.random_seed)
    # create_pretrain_data_step1(train_file=args.input_file1,
    #                            save_conversation_file=args.whole_conversation_file,
    #                            save_persona_file=args.uniq_persona_file)
    create_pretrain_data_step2(args)
