import os
import random
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


class InputExample:
    def __init__(self, guid=None, context=None, response=None, personas=None, label=None):
        self.guid = guid
        self.context = context
        self.response = response
        self.personas = personas
        self.label = label


class InputFeatures:
    def __init__(self, context_input_ids=None, context_attention_mask=None, context_token_type_ids=None,
                 response_input_ids=None, response_attention_mask=None, response_token_type_ids=None,
                 personas_input_ids=None, personas_attention_mask=None, personas_token_type_ids=None,
                 label=None):
        self.context_input_ids = context_input_ids
        self.context_attention_mask = context_attention_mask
        self.context_token_type_ids = context_token_type_ids

        self.response_input_ids = response_input_ids
        self.response_attention_mask = response_attention_mask
        self.response_token_type_ids = response_token_type_ids

        self.personas_input_ids = personas_input_ids
        self.personas_attention_mask = personas_attention_mask
        self.personas_token_type_ids = personas_token_type_ids

        self.label = label


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    # evaluate 和 test 不能两个都是true
    assert (evaluate is False or test is False)
    print("task name : ", task)
    # 如 cached_train_bert_personachat_self_original_neg1
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'valid' if evaluate else 'train',
        str(args.model_type),
        str(task),
        str(args.data_suffix)
    ))
    if test:
        print("load test data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'test',
            str(args.model_type),
            str(task),
            str(args.data_suffix)
        ))

    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        examples = []
        if not evaluate and not test:  # 训练数据
            print("Creating features from dataset file at %s", args.train_data_dir)
            with open(args.train_data_dir, "r", encoding="utf-8") as f:
                for first_id, line in enumerate(f):
                    line = line.strip()
                    fields = line.split('\t')
                    context = fields[0]
                    context = context[:-6].split(' _eos_ ')  # 去掉最后一个' _eos_ ' 切分
                    candidates = fields[1].split("|")  # 切分
                    assert len(candidates) == 20
                    label = int(fields[2])  # str->int
                    if "cmu" in task:
                        personas = fields[3].split('|')  # 切分
                    else:
                        personas = fields[4].split('|')  # 切分
                    pos_cand = candidates[label]

                    # 不要所有的负样本，直接保留正样本
                    new_cands = [[pos_cand, "1"]]

                    if first_id < 5:
                        print("{}  data example:".format(task))
                        print("context: ", context)
                        print("personas: ", personas)
                        print("pos cand: ", pos_cand)
                        print("new cand: ", new_cands)

                    for second_id, one_new_cand in enumerate(new_cands):
                        guid = "%s-%s" % (first_id, second_id)
                        examples.append(
                            InputExample(guid=guid, context=context, response=one_new_cand[0], personas=personas,
                                         label=one_new_cand[1]))
        else:
            if test:
                load_file = args.test_data_dir
            else:
                assert evaluate is True
                load_file = args.valid_data_dir
            print("Creating features from dataset file at %s", load_file)
            with open(load_file, "r", encoding="utf-8") as f:
                for first_id, line in enumerate(f):
                    line = line.strip()
                    fields = line.split('\t')
                    context = fields[0]
                    context = context[:-6].split(' _eos_ ')  # 去掉最后一个' _eos_ ' 切分
                    candidates = fields[1].split("|")  # 切分
                    assert len(candidates) == 20
                    label = int(fields[2])  # str->int
                    if "cmu" in task:
                        personas = fields[3].split('|')  # 切分
                    else:
                        personas = fields[4].split('|')  # 切分

                    # 新版构造数据
                    # 直接把正确回复放在第一位
                    # 因为在计算指标的时候假设第一个是正样本
                    pos_cand = candidates[label]
                    neg_cands = candidates[:label] + candidates[label + 1:]
                    new_candidates = [pos_cand] + neg_cands
                    assert pos_cand == new_candidates[0]
                    assert len(new_candidates) == 20
                    if first_id < 5:
                        print("{} train data example:".format(task))
                        print("context: ", context)
                        print("personas: ", personas)
                        print("new_candidates: ", new_candidates)
                        print("label: ", 0)

                    guid = str(first_id)
                    examples.append(
                        InputExample(guid=guid, context=context, response=new_candidates, personas=personas,
                                     label="0"))

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                args=args,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                evaluate=False if (evaluate is False and test is False) else True)

        torch.save(features, cached_features_file)

    all_context_input_ids = torch.tensor([f.context_input_ids for f in features], dtype=torch.long)
    all_context_attention_mask = torch.tensor([f.context_attention_mask for f in features], dtype=torch.long)
    all_context_token_type_ids = torch.tensor([f.context_token_type_ids for f in features], dtype=torch.long)

    all_response_input_ids = torch.tensor([f.response_input_ids for f in features], dtype=torch.long)
    all_response_attention_mask = torch.tensor([f.response_attention_mask for f in features], dtype=torch.long)
    all_response_token_type_ids = torch.tensor([f.response_token_type_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_context_input_ids, all_context_attention_mask, all_context_token_type_ids,
                            all_response_input_ids, all_response_attention_mask, all_response_token_type_ids,
                            all_labels)

    return dataset


def convert_examples_to_features(examples, tokenizer,
                                 args=None,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True, evaluate=False):
    print("is evaluate????????????????????????: ", evaluate)
    print("examples length: ", len(examples))

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 100000 == 0:
            print("Writing example %d" % (ex_index))
        context = example.context
        context_tokens_list = []
        context_token_type_ids = []
        token_type = 0
        for one_c in context:
            t_one_c = tokenizer.tokenize(one_c)
            context_tokens_list += t_one_c
            context_token_type_ids += ([token_type] * len(t_one_c))
        context_tokens_list = tokenizer.convert_tokens_to_ids(context_tokens_list)
        context_tokens_list = context_tokens_list + [tokenizer.convert_tokens_to_ids("[SEP]")]  # 最后加个SEP
        context_token_type_ids += [0]
        assert len(context_tokens_list) == len(context_token_type_ids)

        if len(context_tokens_list) > args.context_max_length - 1:  # -1是因为需要加CLS
            context_tokens_list = context_tokens_list[-(args.context_max_length - 1):]
            context_token_type_ids = context_token_type_ids[-(args.context_max_length - 1):]

        context_tokens_list = [tokenizer.convert_tokens_to_ids("[CLS]")] + context_tokens_list
        context_token_type_ids = [0] + context_token_type_ids
        assert len(context_tokens_list) == len(context_token_type_ids)
        assert len(context_tokens_list) <= args.context_max_length

        context_attention_mask = [1] * len(context_tokens_list)
        # ========================================================================================
        personas = example.personas
        personas_tokens_list = []
        personas_token_type_ids_list = []
        for one_p in personas:
            t_one_p = tokenizer.tokenize(one_p)
            personas_tokens_list += t_one_p
            personas_token_type_ids_list += ([token_type] * len(t_one_p))
        personas_tokens_list = tokenizer.convert_tokens_to_ids(personas_tokens_list)
        personas_tokens_list = personas_tokens_list + [tokenizer.convert_tokens_to_ids("[SEP]")]  # 最后加个SEP
        personas_token_type_ids_list += [0]
        assert len(personas_tokens_list) == len(personas_token_type_ids_list)

        # 如果context 还有剩余的长度，把他给persona
        if len(personas_tokens_list) > args.persona_max_length + args.context_max_length - len(context_tokens_list):
            # persona 截取前面的
            cut_length = args.persona_max_length + args.context_max_length - len(context_tokens_list)
            personas_tokens_list = personas_tokens_list[:cut_length]
            personas_token_type_ids_list = personas_token_type_ids_list[:cut_length]
        personas_attention_mask_list = [1] * len(personas_tokens_list)

        # 将两者+起来
        context_tokens_list = context_tokens_list + personas_tokens_list
        context_token_type_ids = context_token_type_ids + personas_token_type_ids_list
        context_attention_mask = context_attention_mask + personas_attention_mask_list

        # padding操作
        if len(context_tokens_list) < args.context_max_length + args.persona_max_length:
            padding_length = args.context_max_length + args.persona_max_length - len(context_tokens_list)
            context_tokens_list = context_tokens_list + [0] * padding_length
            context_token_type_ids = context_token_type_ids + [0] * padding_length
            context_attention_mask = context_attention_mask + [0] * padding_length

        assert len(context_tokens_list) == args.context_max_length + args.persona_max_length
        assert len(context_token_type_ids) == args.context_max_length + args.persona_max_length
        assert len(context_attention_mask) == args.context_max_length + args.persona_max_length

        # ========================================================================================
        if not evaluate:
            response = example.response
            response_tokens = tokenizer.tokenize(response)
            if len(response_tokens) > args.response_max_length - 2:  # -2 是因为有[CLS] [SEP]
                response_tokens = response_tokens[:args.response_max_length - 2]
            response_tokens = ["[CLS]"] + response_tokens + ["[SEP]"]
            response_tokens = tokenizer.convert_tokens_to_ids(response_tokens)

            response_attention_mask = [1] * len(response_tokens)
            if len(response_tokens) < args.response_max_length:
                padding_length = args.response_max_length - len(response_tokens)
                response_tokens = response_tokens + [0] * padding_length
                response_attention_mask = response_attention_mask + [0] * padding_length
            response_token_type_ids = [0] * args.response_max_length

            assert len(response_tokens) == args.response_max_length
            assert len(response_attention_mask) == args.response_max_length
            assert len(response_token_type_ids) == args.response_max_length
        else:
            # 这里的response已经变成了list
            response_list = example.response
            assert len(response_list) == 20
            response_tokens_list = []
            response_attention_mask_list = []
            response_token_type_ids_list = []
            for response in response_list:
                response_tokens = tokenizer.tokenize(response)
                if len(response_tokens) > args.response_max_length - 2:  # -2 是因为有[CLS] [SEP]
                    response_tokens = response_tokens[:args.response_max_length - 2]
                response_tokens = ["[CLS]"] + response_tokens + ["[SEP]"]
                response_tokens = tokenizer.convert_tokens_to_ids(response_tokens)

                response_attention_mask = [1] * len(response_tokens)
                if len(response_tokens) < args.response_max_length:
                    padding_length = args.response_max_length - len(response_tokens)
                    response_tokens = response_tokens + [0] * padding_length
                    response_attention_mask = response_attention_mask + [0] * padding_length
                response_token_type_ids = [0] * args.response_max_length

                assert len(response_tokens) == args.response_max_length
                assert len(response_attention_mask) == args.response_max_length
                assert len(response_token_type_ids) == args.response_max_length

                response_tokens_list.append(response_tokens)
                response_attention_mask_list.append(response_attention_mask)
                response_token_type_ids_list.append(response_token_type_ids)

        # ===============================================================================
        if ex_index < 5:
            print("*** Example ***")
            print("example.context: %s " % (example.context))
            print("example.personas: %s " % (example.personas))
            print("context_tokens_list: %s " % (context_tokens_list))
            print("context_attention_mask: %s " % (context_attention_mask))
            print("context_token_type_ids: %s " % (context_token_type_ids))

            print("example.response: %s " % (example.response))
            if not evaluate:
                print("response_tokens: %s " % (response_tokens))
                print("response_attention_mask: %s " % (response_attention_mask))
                print("response_token_type_ids: %s " % (response_token_type_ids))
            else:
                print("response_tokens_list: %s " % (response_tokens_list))
                print("response_attention_mask_list: %s " % (response_attention_mask_list))
                print("response_token_type_ids_list: %s " % (response_token_type_ids_list))

            print("label: ", example.label)  # 其实train的都是1 valid都是0

        if not evaluate:
            features.append(InputFeatures(
                context_input_ids=context_tokens_list,
                context_attention_mask=context_attention_mask,
                context_token_type_ids=context_token_type_ids,

                response_input_ids=response_tokens,
                response_attention_mask=response_attention_mask,
                response_token_type_ids=response_token_type_ids,

                label=int(example.label)))
        else:
            features.append(InputFeatures(
                context_input_ids=context_tokens_list,
                context_attention_mask=context_attention_mask,
                context_token_type_ids=context_token_type_ids,

                response_input_ids=response_tokens_list,
                # [neg_num+1, response_max_length] or [20, response_max_length]
                response_attention_mask=response_attention_mask_list,
                response_token_type_ids=response_token_type_ids_list,

                label=int(example.label)))

    return features


def mean_average_precision(preds_list, grouth_true_list):
    # preds_list和grouth_true_list都是numpy的结构
    print("============= mean average precision ===============")
    assert preds_list.shape[0] == grouth_true_list.shape[0]
    assert preds_list.shape[0] % 20 == 0
    preds_list = preds_list.tolist()
    grouth_true_list = grouth_true_list.tolist()
    # print(preds_list[0:21])
    # print(grouth_true_list[0:21])

    num_query = 0
    map = 0.0
    # 每20个当成一个query
    for index in range(len(preds_list)):
        if index % 20 == 0:
            assert grouth_true_list[index] == 1
            num_query += 1
            # 20个作为一组
            one_preds_list = preds_list[index:index + 20]
            one_grouth_true_list = grouth_true_list[index:index + 20]
            # 这里要把grouth true加入到preds_list,为后续算map做准备
            # 让one_preds_list 从 [[0.7],[0.4],.....] --> [[0.7,1],[0.4,0],.....]
            for j in range(len(one_preds_list)):
                one_preds_list[j].append(one_grouth_true_list[j])
            # 然后需要排序
            sorted_one_preds_list = sorted(one_preds_list, key=lambda x: x[0], reverse=True)
            num_relevant_doc = 0.0
            avp = 0.0
            for j in range(len(sorted_one_preds_list)):
                label = sorted_one_preds_list[j][1]
                if label == 1:
                    num_relevant_doc += 1
                    precision = num_relevant_doc / (j + 1)
                    avp += precision
            avp = avp / num_relevant_doc
            map += avp

    if num_query == 0:
        return 0.0
    else:
        map = map / num_query
        return map


def mean_reciprocal_rank(preds_list, grouth_true_list):
    # preds_list和grouth_true_list都是numpy的结构
    print("============= mean reciprocal rank ===============")
    assert preds_list.shape[0] == grouth_true_list.shape[0]
    assert preds_list.shape[0] % 20 == 0
    preds_list = preds_list.tolist()
    grouth_true_list = grouth_true_list.tolist()
    # print(preds_list[0:21])
    # print(grouth_true_list[0:21])

    num_query = 0
    mrr = 0.0

    for index in range(len(preds_list)):
        if index % 20 == 0:
            assert grouth_true_list[index] == 1
            num_query += 1
            # 20个作为一组
            one_preds_list = preds_list[index:index + 20]
            one_grouth_true_list = grouth_true_list[index:index + 20]
            # 这里要把grouth true加入到preds_list,为后续算map做准备
            # 让one_preds_list 从 [[0.7],[0.4],.....] --> [[0.7,1],[0.4,0],.....]
            for j in range(len(one_preds_list)):
                one_preds_list[j].append(one_grouth_true_list[j])
            # 然后需要排序
            sorted_one_preds_list = sorted(one_preds_list, key=lambda x: x[0], reverse=True)
            for j in range(len(sorted_one_preds_list)):
                label = sorted_one_preds_list[j][1]
                if label == 1:
                    mrr += 1.0 / (j + 1)
                    break

    if num_query == 0:
        return 0.0
    else:
        mrr = mrr / num_query
        return mrr


def top_1_precision(preds_list, grouth_true_list):
    # preds_list和grouth_true_list都是numpy的结构
    print("============= top_1_precision ===============")
    assert preds_list.shape[0] == grouth_true_list.shape[0]
    assert preds_list.shape[0] % 20 == 0
    preds_list = preds_list.tolist()
    grouth_true_list = grouth_true_list.tolist()
    print(preds_list[0:20])
    print(grouth_true_list[0:20])
    print(preds_list[20:40])
    print(grouth_true_list[20:40])
    print(preds_list[120:140])
    print(grouth_true_list[120:140])

    num_query = 0
    top_1_correct = 0.0
    top_2_correct = 0.0
    top_5_correct = 0.0

    for index in range(len(preds_list)):
        if index % 20 == 0:
            assert grouth_true_list[index] == 1
            num_query += 1
            # 20个作为一组
            one_preds_list = preds_list[index:index + 20]
            one_grouth_true_list = grouth_true_list[index:index + 20]
            # 这里要把grouth true加入到preds_list,为后续算map做准备
            # 让one_preds_list 从 [[0.7],[0.4],.....] --> [[0.7,1],[0.4,0],.....]
            for j in range(len(one_preds_list)):
                one_preds_list[j].append(one_grouth_true_list[j])
            # 然后需要排序
            sorted_one_preds_list = sorted(one_preds_list, key=lambda x: x[0], reverse=True)

            # 取排序后第一个，看label是否为1
            label = sorted_one_preds_list[0][1]
            if label == 1:
                top_1_correct += 1

            # 看前两个
            for tmp in sorted_one_preds_list[:2]:
                if tmp[1] == 1:
                    top_2_correct += 1

            # 看前五个
            for tmp in sorted_one_preds_list[:5]:
                if tmp[1] == 1:
                    top_5_correct += 1

    if num_query == 0:
        return 0.0, 0.0, 0.0
    else:
        top_1_correct = top_1_correct / num_query
        top_2_correct = top_2_correct / num_query
        top_5_correct = top_5_correct / num_query
        return top_1_correct, top_2_correct, top_5_correct
