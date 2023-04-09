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


def create_pretrain_data_step1(train_file=None, valid_file=None, save_conversation_file=None, save_persona_file=None):
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

    all_valid_dial = []
    one_dial = []
    with open(valid_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.split()[0] == "1":
                all_valid_dial.append(one_dial)
                one_dial = []
            one_dial.append(line)
        all_valid_dial.append(one_dial)
        all_valid_dial.remove([])

    print("{} is composed of {} dialogues".format(train_file, len(all_train_dial)))
    print("{} is composed of {} dialogues".format(valid_file, len(all_valid_dial)))

    all_dial = all_train_dial + all_valid_dial
    print("train + valid data {}".format(len(all_dial)))

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
    count = 0
    for one in conversation_len_list:
        if one <= args.max_conversation_length:
            count += 1
    print("less than max_conversation_length rate: ", count / len(conversation_and_persona))

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
    print("max len ", max(persona_len_list))
    print("min len ", min(persona_len_list))

    # ==============================================================================
    # personachat数据集 persona数量最少3 最多5
    shuffle_conversation_and_persona = []
    for one_c_and_p in conversation_and_persona:
        tmp = one_c_and_p[1]
        shuffle_list = [tmp]

        tmp = shuffle_list[-1]
        # 打乱次数要动态决定
        if len(tmp.split("|")) == 3:
            shuffle_num = 4
            print(tmp)
        else:
            shuffle_num = 4 + 5

        for _ in range(shuffle_num):
            while tmp in shuffle_list:
                tmp = tmp.split("|")
                random.shuffle(tmp)
                tmp = "|".join(tmp)
            shuffle_list.append(tmp)
            shuffle_conversation_and_persona.append([one_c_and_p[0], tmp, "1"])
    # assert len(shuffle_conversation_and_persona) == len(conversation_and_persona) * shuffle_num

    neg_conversation_and_persona = []
    neg_num = 5 + 5
    for one_c_and_p in conversation_and_persona:
        assert one_c_and_p[1] in uniq_personas
        neg_list = []
        for _ in range(neg_num):
            tmp = random.choice(uniq_personas)
            while tmp == one_c_and_p[1] or tmp in neg_list:
                tmp = random.choice(uniq_personas)
            neg_list.append(tmp)
            neg_conversation_and_persona.append([one_c_and_p[0], tmp, "0"])

    conversation_and_persona = conversation_and_persona + shuffle_conversation_and_persona + neg_conversation_and_persona
    print("shuffle_conversation_and_persona length {}".format(len(shuffle_conversation_and_persona)))
    print("conversation_and_persona neg_conversation_and_persona {}".format(len(neg_conversation_and_persona)))
    print("conversation_and_persona length {}".format(len(conversation_and_persona)))

    with open(save_conversation_file, "w", encoding="utf-8") as file:
        for one_c_and_p in conversation_and_persona:
            assert len(one_c_and_p) == 3
            file.write("\t".join(one_c_and_p) + "\n")

    with open(save_persona_file, "w", encoding="utf-8") as file:
        for one_persona in uniq_personas:
            file.write(one_persona + "\n")


def create_pretrain_data_step1_cmudog(train_file=None, valid_file=None, save_conversation_file=None,
                                      save_persona_file=None):
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

    all_valid_dial = []
    one_dial = []
    with open(valid_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.split()[0] == "1":
                all_valid_dial.append(one_dial)
                one_dial = []
            one_dial.append(line)
        all_valid_dial.append(one_dial)
        all_valid_dial.remove([])

    print("{} is composed of {} dialogues".format(train_file, len(all_train_dial)))
    print("{} is composed of {} dialogues".format(valid_file, len(all_valid_dial)))

    all_dial = all_train_dial + all_valid_dial
    print("train + valid data {}".format(len(all_dial)))

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

        # cmudog 的persona 不需要排序 直接用就可以
        persona = "|".join(persona)
        if persona not in uniq_personas:
            uniq_personas.append(persona)

        # 去掉对话中最大和最小的
        tmp_len = 0
        for one_c in context_history:
            tmp_len += len(one_c.split(" "))
        if tmp_len == 12848 or tmp_len == 22:
            continue
        conversation_and_persona.append([" _eos_ ".join(context_history), persona, "1"])

    print("conversation_and_persona length {}".format(len(conversation_and_persona)))
    print("uniq_personas length {}".format(len(uniq_personas)))

    # ================================================================================
    # 统计下每个对话长度
    conversation_len_list = []
    conversation_turn_list = []
    each_sentence_list = []
    for index, one_c_and_p in enumerate(conversation_and_persona):
        if index < 5:
            print(one_c_and_p)
        one_c = one_c_and_p[0].split(" _eos_ ")
        oc_len = 0
        for oc in one_c:
            oc_len += len(oc.split(" "))
            each_sentence_list.append(len(oc.split(" ")))
        conversation_len_list.append(oc_len)
        conversation_turn_list.append(len(one_c))
        if conversation_len_list[-1] == 12848 or conversation_len_list[-1] == 22:
            print(conversation_len_list[-1])
            print(one_c)

    print("conversation_len_list: ", sorted(conversation_len_list, reverse=True))
    print("avg conversation_len: ", sum(conversation_len_list) / len(conversation_len_list))
    print("conversation_len_list count: ", Counter(conversation_len_list))
    print("max", max(conversation_len_list))
    print("min", min(conversation_len_list))
    print("=" * 50)
    print("conversation_turn_list: ", sorted(conversation_turn_list, reverse=True))
    print("avg conversation_turn: ", sum(conversation_turn_list) / len(conversation_turn_list))
    print("conversation_turn_list count: ", Counter(conversation_turn_list))
    print("avg each_sentence: ", sum(each_sentence_list) / len(each_sentence_list))
    print("each_sentence_list count: ", Counter(each_sentence_list))

    count = 0
    for one in conversation_len_list:
        if one <= args.max_conversation_length:
            count += 1
    print("less than max_conversation_length rate: ", count / len(conversation_and_persona))

    # 统计persona长度
    persona_len_list = []
    persona_turn_list = []
    each_persona_list = []
    for index, one_p in enumerate(uniq_personas):
        one_p = one_p.split("|")
        op_len = 0
        for op in one_p:
            op_len += len(op.split(" "))
            each_persona_list.append(len(op.split(" ")))
        persona_len_list.append(op_len)
        persona_turn_list.append(len(one_p))

    print("persona_len_list: ", sorted(persona_len_list, reverse=True))
    print("avg persona_len: ", sum(persona_len_list) / len(persona_len_list))
    print("persona_len_list count: ", Counter(persona_len_list))
    print("max len ", max(persona_len_list))
    print("min len ", min(persona_len_list))
    print("=" * 50)
    print("persona_turn_list: ", sorted(persona_turn_list, reverse=True))
    print("avg persona_turn: ", sum(persona_turn_list) / len(persona_turn_list))
    print("persona_turn_list count: ", Counter(persona_turn_list))
    print("avg each_persona: ", sum(each_persona_list) / len(each_persona_list))
    print("each_persona_list count: ", Counter(each_persona_list))

    # ==============================================================================
    # 因为conversation 和 persona句子都太长了，切块
    pos_conversation_and_persona = []
    for index, one_c_and_p in enumerate(conversation_and_persona):
        one_c = one_c_and_p[0].split(" _eos_ ")
        one_p = one_c_and_p[1].split("|")

        new_c_list = []
        new_u_str = ""
        for u_str in one_c:
            if len(new_u_str.split(" ")) + len(u_str.split(" ")) <= args.max_conversation_length:
                new_u_str = new_u_str + u_str + " _eos_ "
            else:
                new_u_str = new_u_str[:-7]  # 去掉最后一个" _eos_ "
                new_c_list.append(new_u_str)
                new_u_str = u_str + " _eos_ "
        # 把最后一个也补进去
        new_u_str = new_u_str[:-7]
        new_c_list.append(new_u_str)

        new_p_list = []
        new_p_str = ""
        for p_str in one_p:
            if len(new_p_str.split(" ")) + len(p_str.split(" ")) <= args.max_personas_length:
                new_p_str = new_p_str + p_str + " | "
            else:
                new_p_str = new_p_str[:-3]
                new_p_list.append(new_p_str)
                new_p_str = p_str + " | "
        new_p_str = new_p_str[:-3]
        new_p_list.append(new_p_str)

        if index < 3:
            print("=" * 50)
            print(one_c)
            print(new_c_list)
            print(len(new_c_list))
            print(one_p)
            print(new_p_list)
            print(len(new_p_list))

        for new_u_str in new_c_list:
            for new_p_str in new_p_list:
                new_p_str = new_p_str.split(" | ")
                new_p_str = "|".join(new_p_str)
                pos_conversation_and_persona.append([new_u_str, new_p_str, "1"])

        # 这里是消融实验用来取消数据增强部分的
        # new_p_str = new_p_list[0].split(" | ")
        # new_p_str = "|".join(new_p_str)
        # pos_conversation_and_persona.append([new_c_list[0], new_p_str, "1"])

    print("neg_conversation_and_persona sample")
    neg_conversation_and_persona = []
    for index, one_c_and_p in enumerate(conversation_and_persona):
        assert one_c_and_p[1] in uniq_personas
        one_c = one_c_and_p[0].split(" _eos_ ")
        one_p = random.choice(uniq_personas)
        while one_p == one_c_and_p[1]:
            one_p = random.choice(uniq_personas)
        one_p = one_p.split("|")

        new_c_list = []
        new_u_str = ""
        for u_str in one_c:
            if len(new_u_str.split(" ")) + len(u_str.split(" ")) <= args.max_conversation_length:
                new_u_str = new_u_str + u_str + " _eos_ "
            else:
                new_u_str = new_u_str[:-7]  # 去掉最后一个" _eos_ "
                new_c_list.append(new_u_str)
                new_u_str = u_str + " _eos_ "
        # 把最后一个也补进去
        new_u_str = new_u_str[:-7]
        new_c_list.append(new_u_str)

        new_p_list = []
        new_p_str = ""
        for p_str in one_p:
            if len(new_p_str.split(" ")) + len(p_str.split(" ")) <= args.max_personas_length:
                new_p_str = new_p_str + p_str + " | "
            else:
                new_p_str = new_p_str[:-3]
                new_p_list.append(new_p_str)
                new_p_str = p_str + " | "
        new_p_str = new_p_str[:-3]
        new_p_list.append(new_p_str)

        if index < 3:
            print("=" * 50)
            print(one_c)
            print(new_c_list)
            print(len(new_c_list))
            print(one_p)
            print(new_p_list)
            print(len(new_p_list))

        for new_u_str in new_c_list:
            for new_p_str in new_p_list:
                new_p_str = new_p_str.split(" | ")
                new_p_str = "|".join(new_p_str)
                neg_conversation_and_persona.append([new_u_str, new_p_str, "0"])

        # 这里是消融实验用来取消数据增强部分的
        # new_p_str = new_p_list[0].split(" | ")
        # new_p_str = "|".join(new_p_str)
        # neg_conversation_and_persona.append([new_c_list[0], new_p_str, "0"])

    conversation_and_persona = pos_conversation_and_persona + neg_conversation_and_persona
    print("pos_conversation_and_persona length {}".format(len(pos_conversation_and_persona)))
    print("neg_conversation_and_persona {}".format(len(neg_conversation_and_persona)))
    print("conversation_and_persona length {}".format(len(conversation_and_persona)))

    with open(save_conversation_file, "w", encoding="utf-8") as file:
        for one_c_and_p in conversation_and_persona:
            assert len(one_c_and_p) == 3
            file.write("\t".join(one_c_and_p) + "\n")

    with open(save_persona_file, "w", encoding="utf-8") as file:
        for one_persona in uniq_personas:
            file.write(one_persona + "\n")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels


class TrainingFeature(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, masked_lm_labels,
                 next_sentence_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.masked_lm_labels = masked_lm_labels
        self.next_sentence_labels = next_sentence_labels


def create_pretrain_data_step2(args=None, rng=None):
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained)
    bert_tokenizer.add_tokens(args.add_tokens)
    print(len(bert_tokenizer))
    vocab_words = list(bert_tokenizer.vocab.keys())
    print(vocab_words)

    all_documents = []
    with open(args.whole_conversation_file, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index % 10000 == 0:
                print(index)
            line = line.strip().split("\t")  # [conversation (" _eos_ " 分割),personas ("|" 分割), label (str类型)]
            assert len(line) == 3
            if index < 5:
                print(line)
            conversations = line[0].split(" _eos_ ")
            personas = line[1].split("|")
            token_conversations = []
            for one in conversations:
                token_conversations.append(bert_tokenizer.tokenize(one))
            token_personas = []
            for one in personas:
                token_personas.append(bert_tokenizer.tokenize(one))

            if index < 5:
                print(token_conversations)
                print(token_personas)

            all_documents.append([token_conversations, token_personas, line[2]])

    print("all_documents length: ", len(all_documents))

    # create_instances_from_document
    instances = []
    for document_index in range(len(all_documents)):
        document = all_documents[document_index]
        if document_index < 5:
            print(document[0])  # token_conversations
            print(document[1])  # token_personas
            print(document[2])

        cut_conversation = []
        for one in document[0]:  # token_conversations
            cut_conversation.extend(one + ["[EOU]"])
        cut_conversation.pop(-1)
        cut_conversation += ["[SEP]"]

        cut_personas = []
        for one in document[1]:  # token_personas
            cut_personas.extend(one + ["[EOK]"])
        cut_personas.pop(-1)
        cut_personas += ["[SEP]"]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        if len(cut_conversation) > args.max_conversation_length:
            cut_conversation = cut_conversation[-args.max_conversation_length:]
            if cut_conversation[0] == "[EOU]":
                cut_conversation = cut_conversation[1:]
        tokens += cut_conversation
        segment_ids += [0] * len(cut_conversation)
        assert len(tokens) == len(segment_ids)

        if len(cut_personas) > args.max_personas_length:
            cut_personas = cut_personas[:args.max_personas_length]
        tokens += cut_personas
        segment_ids += [1] * len(cut_personas)
        assert len(tokens) == len(segment_ids)

        (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(args,
                                                                                       tokens, args.masked_lm_prob,
                                                                                       args.max_predictions_per_seq,
                                                                                       vocab_words, rng)

        assert document[2] == "1" or document[2] == "0"
        if document[2] == "1":
            is_random_next = False
        else:
            is_random_next = True

        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        if document_index < 5:
            print("TrainingInstance: ")
            print(instance.tokens)
            print(instance.segment_ids)
            print(instance.is_random_next)
            print(instance.masked_lm_positions)
            print(instance.masked_lm_labels)
        instances.append(instance)

    # ===============================================================================
    features = []

    for index, instance in enumerate(instances):
        input_ids = bert_tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= args.max_seq_length

        while len(input_ids) < args.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = bert_tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        assert len(masked_lm_positions) <= args.max_predictions_per_seq and len(masked_lm_positions) == len(
            masked_lm_ids)

        # masked_lm_positions masked_lm_ids 只是中间产物，后面并不需要
        anno_masked_lm_labels = [-1] * args.max_seq_length
        for pos, label in zip(masked_lm_positions, masked_lm_ids):
            anno_masked_lm_labels[pos] = label

        next_sentence_label = 0 if instance.is_random_next else 1

        features.append(TrainingFeature(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            masked_lm_labels=anno_masked_lm_labels,
            next_sentence_labels=next_sentence_label
        ))

        if index < 5:
            print("TrainingFeature: ")
            print(features[-1].input_ids)
            print(features[-1].attention_mask)
            print(features[-1].token_type_ids)
            print(features[-1].masked_lm_labels)
            print(features[-1].next_sentence_labels)
            print(masked_lm_positions)
            print(masked_lm_ids)

    torch.save(features, args.output_file)


class Args:
    def __init__(self):
        # 这里把train和valid数据都用在预训练上
        self.task_name = "cmudog"

        if self.task_name == "personachat_self_original":
            self.input_file1 = "/data/mgw_data/FIRE_data/personachat_data/personachat/train_self_original.txt"
            self.input_file2 = "/data/mgw_data/FIRE_data/personachat_data/personachat/valid_self_original.txt"
            # 分两步处理，第一步把train和valid里面完整的conversation和persona拿出来,
            # 然后所有独立的persona也拿出来，用来构造负样本
            self.whole_conversation_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/personachat_self_original_conversation.txt"
            self.uniq_persona_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/personachat_self_original_persona.txt"
            # 第二步才是最终的预训练数据
            self.output_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/personachat_self_original_pretraining_data_V2.pkl"
        elif self.task_name == "personachat_self_revised":
            self.input_file1 = "/data/mgw_data/FIRE_data/personachat_data/personachat/train_self_revised.txt"
            self.input_file2 = "/data/mgw_data/FIRE_data/personachat_data/personachat/valid_self_revised.txt"
            self.whole_conversation_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/personachat_self_revised_conversation.txt"
            self.uniq_persona_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/personachat_self_revised_persona.txt"
            self.output_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/personachat_self_revised_pretraining_data_V2.pkl"
        elif self.task_name == "cmudog":
            self.input_file1 = "/data/mgw_data/FIRE_data/cmudog_data/cmudog/train_self_original_fullSection.txt"
            self.input_file2 = "/data/mgw_data/FIRE_data/cmudog_data/cmudog/valid_self_original_fullSection.txt"
            self.whole_conversation_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/cmudog_conversation.txt"
            self.uniq_persona_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/cmudog_persona.txt"
            self.output_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/cmudog_pretraining_data_V2.pkl"
            # 这里做了个消融实验，取消数据增强部分
            # self.whole_conversation_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/cmudog_conversation_noDA.txt"
            # self.uniq_persona_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/cmudog_persona_noDA.txt"
            # self.output_file = "/data/mgw_data/BERT_persona_response_selection/pretraining_data/cmudog_pretraining_data_V2noDA.pkl"
        else:
            raise NotImplementedError

        self.bert_pretrained = "/data/mgw_data/bert_base_uncased_model"
        self.do_lower_case = True
        self.do_whole_word_mask = True
        self.max_conversation_length = 400  # 330  # 算上[SEP] 和 [EOU]
        self.max_personas_length = 100  # 100  # 算上[SEP] 和 [EOK]
        self.max_seq_length = self.max_conversation_length + self.max_personas_length + 1  # 这里是conversation+personas的总长度，+1是算是[CLS]

        self.add_tokens = ['[EOU]', '[EOK]']
        self.max_predictions_per_seq = round(self.max_seq_length * 0.15)  # 大概是max_seq_length*0.15
        self.random_seed = 2022
        self.dupe_factor = 5  # Number of times to duplicate the input data (with different masks).
        self.masked_lm_prob = 0.15
        self.short_seq_prob = 0.1  # Probability of creating sequences which are shorter than the maximum length.
        self.special_tok = None  # 一般utterance之间会用[EOT]隔开，这里就直接用[SEP]隔开好了


if __name__ == "__main__":
    args = Args()
    rng = random.Random(args.random_seed)
    if args.task_name == "cmudog":
        # 这个数据集的persona太长了，另外处理
        create_pretrain_data_step1_cmudog(train_file=args.input_file1, valid_file=args.input_file2,
                                          save_conversation_file=args.whole_conversation_file,
                                          save_persona_file=args.uniq_persona_file)
    else:
        create_pretrain_data_step1(train_file=args.input_file1, valid_file=args.input_file2,
                                   save_conversation_file=args.whole_conversation_file,
                                   save_persona_file=args.uniq_persona_file)
    create_pretrain_data_step2(args, rng)
