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
                    '''
                    # 这里都是旧版的，如果需要用替换这部分就好
                    pos_cand = candidates[label]
                    neg_cands = candidates[:label] + candidates[label + 1:]
                    if pos_cand in neg_cands:
                        print("pos cand in neg_cands: ", first_id)

                    # 直接把正确回复放在第一位
                    new_cands = [[pos_cand, "1"]]
                    for one_neg_cand in neg_cands:
                        new_cands.append([one_neg_cand, "0"])
                    assert len(new_cands) == 20

                    if first_id < 5:
                        print("{} train data example:".format(task))
                        print("context: ", context)
                        print("personas: ", personas)
                        print("pos cand: ", pos_cand)
                        print("new cand: ", new_cands)

                    for second_id, one_new_cand in enumerate(new_cands):
                        guid = "%s-%s" % (first_id, second_id)
                        examples.append(
                            InputExample(guid=guid, context=context, response=one_new_cand[0], personas=personas,
                                         label=one_new_cand[1]))
                    '''
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

    all_personas_input_ids = torch.tensor([f.personas_input_ids for f in features], dtype=torch.long)
    all_personas_attention_mask = torch.tensor([f.personas_attention_mask for f in features], dtype=torch.long)
    all_personas_token_type_ids = torch.tensor([f.personas_token_type_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_context_input_ids, all_context_attention_mask, all_context_token_type_ids,
                            all_response_input_ids, all_response_attention_mask, all_response_token_type_ids,
                            all_personas_input_ids, all_personas_attention_mask, all_personas_token_type_ids,
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
            t_one_c = tokenizer.tokenize(one_c) + ["[EOU]"]
            context_tokens_list += t_one_c
            context_token_type_ids += ([token_type] * len(t_one_c))
            token_type = int(1 - token_type)
        context_tokens_list = tokenizer.convert_tokens_to_ids(context_tokens_list)
        context_tokens_list.pop(-1)
        context_tokens_list = context_tokens_list + [tokenizer.convert_tokens_to_ids("[SEP]")]  # 最后一个换成SEP
        assert len(context_tokens_list) == len(context_token_type_ids)

        if len(context_tokens_list) > args.context_max_length - 1:  # -1是因为需要加CLS
            context_tokens_list = context_tokens_list[-(args.context_max_length - 1):]
            context_token_type_ids = context_token_type_ids[-(args.context_max_length - 1):]
            if context_tokens_list[0] == tokenizer.convert_tokens_to_ids("[EOU]"):
                context_tokens_list = context_tokens_list[1:]
                context_token_type_ids = context_token_type_ids[1:]

        context_tokens_list = [tokenizer.convert_tokens_to_ids("[CLS]")] + context_tokens_list
        context_token_type_ids = [0] + context_token_type_ids
        assert len(context_tokens_list) == len(context_token_type_ids)

        context_attention_mask = [1] * len(context_tokens_list)
        # padding操作
        if len(context_tokens_list) < args.context_max_length:
            padding_length = args.context_max_length - len(context_tokens_list)
            context_tokens_list = context_tokens_list + [0] * padding_length
            context_token_type_ids = context_token_type_ids + [0] * padding_length
            context_attention_mask = context_attention_mask + [0] * padding_length

        assert len(context_tokens_list) == args.context_max_length
        assert len(context_token_type_ids) == args.context_max_length
        assert len(context_attention_mask) == args.context_max_length

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

        # ========================================================================================
        personas = example.personas
        if len(personas) > args.max_persona_num:
            personas = personas[-args.max_persona_num:]
        personas_tokens_list = []
        personas_attention_mask_list = []
        personas_token_type_ids_list = []
        for one_p in personas:
            t_one_p = tokenizer.tokenize(one_p)
            if len(t_one_p) > args.persona_max_length - 2:
                t_one_p = t_one_p[:args.persona_max_length - 2]
            t_one_p = ["[CLS]"] + t_one_p + ["[SEP]"]
            t_one_p = tokenizer.convert_tokens_to_ids(t_one_p)

            one_atten_mask = [1] * len(t_one_p)
            if len(t_one_p) < args.persona_max_length:
                padding_length = args.persona_max_length - len(t_one_p)
                t_one_p = t_one_p + [0] * padding_length
                one_atten_mask = one_atten_mask + [0] * padding_length
            one_token_type_ids = [0] * args.persona_max_length

            assert len(t_one_p) == args.persona_max_length
            assert len(one_atten_mask) == args.persona_max_length
            assert len(one_token_type_ids) == args.persona_max_length

            personas_tokens_list.append(t_one_p)
            personas_attention_mask_list.append(one_atten_mask)
            personas_token_type_ids_list.append(one_token_type_ids)

        if len(personas_tokens_list) < args.max_persona_num:
            for i in range(args.max_persona_num - len(personas_tokens_list)):
                personas_tokens_list.append([0] * args.persona_max_length)
                personas_attention_mask_list.append([0] * args.persona_max_length)
                personas_token_type_ids_list.append([0] * args.persona_max_length)

        assert len(personas_tokens_list) == args.max_persona_num and \
               len(personas_tokens_list[0]) == args.persona_max_length
        assert len(personas_attention_mask_list) == args.max_persona_num and \
               len(personas_attention_mask_list[0]) == args.persona_max_length
        assert len(personas_token_type_ids_list) == args.max_persona_num and \
               len(personas_token_type_ids_list[0]) == args.persona_max_length

        # ===============================================================================
        if ex_index < 5:
            print("*** Example ***")
            print("example.context: %s " % (example.context))
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

            print("example.personas: %s " % (example.personas))
            print("personas_tokens_list: %s " % (personas_tokens_list))
            print("personas_attention_mask_list: %s " % (personas_attention_mask_list))
            print("personas_token_type_ids_list: %s " % (personas_token_type_ids_list))

            print("label: ", example.label)  # 其实train的都是1 valid都是0

        if not evaluate:
            features.append(InputFeatures(
                context_input_ids=context_tokens_list,
                context_attention_mask=context_attention_mask,
                context_token_type_ids=context_token_type_ids,

                response_input_ids=response_tokens,
                response_attention_mask=response_attention_mask,
                response_token_type_ids=response_token_type_ids,

                personas_input_ids=personas_tokens_list,  # [max_persona_num, persona_max_length]
                personas_attention_mask=personas_attention_mask_list,
                personas_token_type_ids=personas_token_type_ids_list,
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

                personas_input_ids=personas_tokens_list,  # [max_persona_num, persona_max_length]
                personas_attention_mask=personas_attention_mask_list,
                personas_token_type_ids=personas_token_type_ids_list,
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


# =======================================================================================================
import collections

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(args, tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpiecesdk
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (args.do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


# ================================================================================================
class InputExample_Multi_Task:
    def __init__(self, guid=None, context=None, response=None, personas=None,
                 uncorrelated_personas=None, uncorrelated_personas_index=None,
                 neg_personas=None, label=None):
        self.guid = guid
        self.context = context
        self.response = response
        self.personas = personas
        self.uncorrelated_personas = uncorrelated_personas
        self.uncorrelated_personas_index = uncorrelated_personas_index
        self.neg_personas = neg_personas
        self.label = label


class InputFeatures_Multi_Task:
    def __init__(self, context_input_ids=None, context_attention_mask=None, context_token_type_ids=None,
                 response_input_ids=None, response_attention_mask=None, response_token_type_ids=None,
                 personas_input_ids=None, personas_attention_mask=None, personas_token_type_ids=None,
                 label=None,
                 uncorrelated_personas_input_ids=None, uncorrelated_personas_attention_mask=None,
                 uncorrelated_personas_token_type_ids=None, uncorrelated_personas_index=None,
                 neg_personas_input_ids=None, neg_personas_attention_mask=None, neg_personas_token_type_ids=None
                 ):
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

        self.uncorrelated_personas_input_ids = uncorrelated_personas_input_ids
        self.uncorrelated_personas_attention_mask = uncorrelated_personas_attention_mask
        self.uncorrelated_personas_token_type_ids = uncorrelated_personas_token_type_ids
        self.uncorrelated_personas_index = uncorrelated_personas_index

        self.neg_personas_input_ids = neg_personas_input_ids
        self.neg_personas_attention_mask = neg_personas_attention_mask
        self.neg_personas_token_type_ids = neg_personas_token_type_ids


def load_and_cache_examples_multi_task(args, task, tokenizer, evaluate=False):
    print("task name : ", task)
    output_mode = "classification"
    # 如 cached_train_bert_personachat_self_original_neg1
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'valid' if evaluate else 'train',
        str(args.model_type),
        str(task),
        str(args.data_suffix)
    ))
    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        label_list = ["0", "1"]
        examples = []
        # 这里读入personas文件
        personas_collection = []
        one_persona_collection = []
        with open(args.personas_file, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                line = line.strip()
                line = line.split("|")
                if index < 5:
                    print(line)
                personas_collection.append(line)
                for one in line:
                    if one not in one_persona_collection:
                        one_persona_collection.append(one)
        print("one_persona_collection len: ", len(one_persona_collection))
        random.shuffle(one_persona_collection)
        print("personas_collection len: ", len(personas_collection))
        print("one_persona_collection len: ", len(one_persona_collection))

        if not evaluate:  # 训练数据
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

                    # 构造多任务的数据
                    # (1)
                    out_persona_index = random.randint(0, len(personas) - 1)
                    tmp_persona = random.choice(one_persona_collection)  # 这里是一个persona
                    while tmp_persona in personas:
                        tmp_persona = random.choice(one_persona_collection)
                    uncorrelated_personas = personas[:out_persona_index] + [tmp_persona] + personas[
                                                                                           out_persona_index + 1:]
                    # (2)
                    tmp_persona = random.choice(personas_collection)  # 这里是一个列表的persona
                    tmp_persona = sorted(tmp_persona)
                    tmp_persona = "|".join(tmp_persona)

                    s_personas = sorted(personas)
                    s_personas = "|".join(s_personas)
                    while tmp_persona == s_personas:
                        tmp_persona = random.choice(personas_collection)  # 这里是一个列表的persona
                        tmp_persona = sorted(tmp_persona)
                        tmp_persona = "|".join(tmp_persona)
                    neg_personas = tmp_persona.split("|")
                    random.shuffle(neg_personas)  # 打乱下,不然因为sorted会一样的

                    if first_id < 5:
                        print("{}  data example:".format(task))
                        print("context: ", context)
                        print("personas: ", personas)
                        print("uncorrelated_personas: ", uncorrelated_personas)
                        print("out_persona_index: ", out_persona_index)  # 这个是int类型
                        print("neg_personas: ", neg_personas)
                        print("pos cand: ", pos_cand)
                        print("new cand: ", new_cands)

                    for second_id, one_new_cand in enumerate(new_cands):
                        guid = "%s-%s" % (first_id, second_id)
                        examples.append(
                            InputExample_Multi_Task(guid=guid, context=context, response=one_new_cand[0],
                                                    personas=personas,
                                                    uncorrelated_personas=uncorrelated_personas,
                                                    uncorrelated_personas_index=out_persona_index,
                                                    neg_personas=neg_personas,
                                                    label=one_new_cand[1]))
        else:
            print("Creating features from dataset file at %s", args.valid_data_dir)
            with open(args.valid_data_dir, "r", encoding="utf-8") as f:
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
                    neg_cands = candidates[:label] + candidates[label + 1:]
                    if pos_cand in neg_cands:
                        print("pos cand in neg_cands: ", first_id)

                    # 直接把正确回复放在第一位
                    new_cands = [[pos_cand, "1"]]
                    for one_neg_cand in neg_cands:
                        new_cands.append([one_neg_cand, "0"])
                    assert len(new_cands) == 20

                    if first_id < 5:
                        print("{} train data example:".format(task))
                        print("context: ", context)
                        print("personas: ", personas)
                        print("pos cand: ", pos_cand)
                        print("new cand: ", new_cands)

                    # valid数据就不用构造多任务的数据了
                    for second_id, one_new_cand in enumerate(new_cands):
                        guid = "%s-%s" % (first_id, second_id)
                        examples.append(
                            InputExample_Multi_Task(guid=guid, context=context, response=one_new_cand[0],
                                                    personas=personas,
                                                    label=one_new_cand[1]))

        features = convert_examples_to_features_multi_task(examples,
                                                           tokenizer,
                                                           label_list=label_list,
                                                           args=args,
                                                           output_mode=output_mode,
                                                           pad_token=
                                                           tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                           pad_token_segment_id=0, evaluate=evaluate)

        torch.save(features, cached_features_file)

    all_context_input_ids = torch.tensor([f.context_input_ids for f in features], dtype=torch.long)
    all_context_attention_mask = torch.tensor([f.context_attention_mask for f in features], dtype=torch.long)
    all_context_token_type_ids = torch.tensor([f.context_token_type_ids for f in features], dtype=torch.long)

    all_response_input_ids = torch.tensor([f.response_input_ids for f in features], dtype=torch.long)
    all_response_attention_mask = torch.tensor([f.response_attention_mask for f in features], dtype=torch.long)
    all_response_token_type_ids = torch.tensor([f.response_token_type_ids for f in features], dtype=torch.long)

    all_personas_input_ids = torch.tensor([f.personas_input_ids for f in features], dtype=torch.long)
    all_personas_attention_mask = torch.tensor([f.personas_attention_mask for f in features], dtype=torch.long)
    all_personas_token_type_ids = torch.tensor([f.personas_token_type_ids for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        raise NotImplementedError

    if not evaluate:
        all_uncorrelated_personas_input_ids = \
            torch.tensor([f.uncorrelated_personas_input_ids for f in features], dtype=torch.long)
        all_uncorrelated_personas_attention_mask = torch.tensor(
            [f.uncorrelated_personas_attention_mask for f in features], dtype=torch.long)
        all_uncorrelated_personas_token_type_ids = torch.tensor(
            [f.uncorrelated_personas_token_type_ids for f in features], dtype=torch.long)
        all_uncorrelated_personas_index = torch.tensor(
            [f.uncorrelated_personas_index for f in features], dtype=torch.long)

        all_neg_personas_input_ids = torch.tensor([f.neg_personas_input_ids for f in features], dtype=torch.long)
        all_neg_personas_attention_mask = torch.tensor([f.neg_personas_attention_mask for f in features],
                                                       dtype=torch.long)
        all_neg_personas_token_type_ids = torch.tensor([f.neg_personas_token_type_ids for f in features],
                                                       dtype=torch.long)

        dataset = TensorDataset(all_context_input_ids, all_context_attention_mask, all_context_token_type_ids,
                                all_response_input_ids, all_response_attention_mask, all_response_token_type_ids,
                                all_personas_input_ids, all_personas_attention_mask, all_personas_token_type_ids,
                                all_labels,
                                all_uncorrelated_personas_input_ids, all_uncorrelated_personas_attention_mask,
                                all_uncorrelated_personas_token_type_ids, all_uncorrelated_personas_index,
                                all_neg_personas_input_ids, all_neg_personas_attention_mask,
                                all_neg_personas_token_type_ids)
    else:

        dataset = TensorDataset(all_context_input_ids, all_context_attention_mask, all_context_token_type_ids,
                                all_response_input_ids, all_response_attention_mask, all_response_token_type_ids,
                                all_personas_input_ids, all_personas_attention_mask, all_personas_token_type_ids,
                                all_labels)

    return dataset


def convert_examples_to_features_multi_task(examples, tokenizer,
                                            args=None,
                                            label_list=None,
                                            output_mode=None,
                                            pad_token=0,
                                            pad_token_segment_id=0,
                                            mask_padding_with_zero=True,
                                            evaluate=None  # 这里加了参数
                                            ):
    label_map = {label: i for i, label in enumerate(label_list)}
    print("label_map: ", label_map)
    print("examples length: ", len(examples))
    print("evaluate mode: ", evaluate)
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 100000 == 0:
            print("Writing example %d" % (ex_index))
        context = example.context
        context_tokens_list = []
        context_token_type_ids = []
        token_type = 0
        for one_c in context:
            t_one_c = tokenizer.tokenize(one_c) + ["[SEP]"]
            context_tokens_list += t_one_c
            context_token_type_ids += ([token_type] * len(t_one_c))
            token_type = int(1 - token_type)
        context_tokens_list = tokenizer.convert_tokens_to_ids(context_tokens_list)
        assert len(context_tokens_list) == len(context_token_type_ids)

        if len(context_tokens_list) > args.context_max_length - 1:  # -1是因为需要加CLS
            context_tokens_list = context_tokens_list[-(args.context_max_length - 1):]
            context_token_type_ids = context_token_type_ids[-(args.context_max_length - 1):]
            if context_tokens_list[0] == tokenizer.convert_tokens_to_ids("[SEP]"):
                context_tokens_list = context_tokens_list[1:]
                context_token_type_ids = context_token_type_ids[1:]

        context_tokens_list = [tokenizer.convert_tokens_to_ids("[CLS]")] + context_tokens_list
        context_token_type_ids = [0] + context_token_type_ids
        assert len(context_tokens_list) == len(context_token_type_ids)

        context_attention_mask = [1] * len(context_tokens_list)
        # padding操作
        if len(context_tokens_list) < args.context_max_length:
            padding_length = args.context_max_length - len(context_tokens_list)
            context_tokens_list = context_tokens_list + [0] * padding_length
            context_token_type_ids = context_token_type_ids + [0] * padding_length
            context_attention_mask = context_attention_mask + [0] * padding_length

        assert len(context_tokens_list) == args.context_max_length
        assert len(context_token_type_ids) == args.context_max_length
        assert len(context_attention_mask) == args.context_max_length

        # ========================================================================================
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

        # ========================================================================================
        personas = example.personas
        if len(personas) > args.max_persona_num:
            personas = personas[-args.max_persona_num:]
        personas_tokens_list = []
        personas_attention_mask_list = []
        personas_token_type_ids_list = []
        for one_p in personas:
            t_one_p = tokenizer.tokenize(one_p)
            if len(t_one_p) > args.persona_max_length - 2:
                t_one_p = t_one_p[:args.persona_max_length - 2]
            t_one_p = ["[CLS]"] + t_one_p + ["[SEP]"]
            t_one_p = tokenizer.convert_tokens_to_ids(t_one_p)

            one_atten_mask = [1] * len(t_one_p)
            if len(t_one_p) < args.persona_max_length:
                padding_length = args.persona_max_length - len(t_one_p)
                t_one_p = t_one_p + [0] * padding_length
                one_atten_mask = one_atten_mask + [0] * padding_length
            one_token_type_ids = [0] * args.persona_max_length

            assert len(t_one_p) == args.persona_max_length
            assert len(one_atten_mask) == args.persona_max_length
            assert len(one_token_type_ids) == args.persona_max_length

            personas_tokens_list.append(t_one_p)
            personas_attention_mask_list.append(one_atten_mask)
            personas_token_type_ids_list.append(one_token_type_ids)

        if len(personas_tokens_list) < args.max_persona_num:
            for i in range(args.max_persona_num - len(personas_tokens_list)):
                personas_tokens_list.append([0] * args.persona_max_length)
                personas_attention_mask_list.append([0] * args.persona_max_length)
                personas_token_type_ids_list.append([0] * args.persona_max_length)

        assert len(personas_tokens_list) == args.max_persona_num and \
               len(personas_tokens_list[0]) == args.persona_max_length
        assert len(personas_attention_mask_list) == args.max_persona_num and \
               len(personas_attention_mask_list[0]) == args.persona_max_length
        assert len(personas_token_type_ids_list) == args.max_persona_num and \
               len(personas_token_type_ids_list[0]) == args.persona_max_length

        # =================================================================================
        # 多任务
        if not evaluate:
            # (1) uncorrelated_persona
            # 这里要防止截取了后，不相关的persona被截取掉了
            uncorrelated_personas = example.uncorrelated_personas
            uncorrelated_personas_index = example.uncorrelated_personas_index
            un_persona = uncorrelated_personas[uncorrelated_personas_index]

            if len(uncorrelated_personas) > args.max_persona_num:
                uncorrelated_personas = uncorrelated_personas[-args.max_persona_num:]
                if un_persona not in uncorrelated_personas:
                    un_index = random.randint(0, len(uncorrelated_personas) - 1)
                    uncorrelated_personas = uncorrelated_personas[:un_index] + [un_persona] + uncorrelated_personas[
                                                                                              un_index + 1:]
                    uncorrelated_personas_index = un_index
                    assert len(uncorrelated_personas) == args.max_persona_num

            uncorrelated_personas_tokens_list, \
            uncorrelated_personas_attention_mask_list, \
            uncorrelated_personas_token_type_ids_list = create_personas_data(args, uncorrelated_personas, tokenizer)

            # (2) neg_persona
            neg_personas = example.neg_personas
            if len(neg_personas) > args.max_persona_num:
                neg_personas = neg_personas[-args.max_persona_num:]
            neg_personas_tokens_list, \
            neg_personas_attention_mask_list, \
            neg_personas_token_type_ids_list = create_personas_data(args, neg_personas, tokenizer)

        if output_mode == "classification":
            label = label_map[example.label]
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            print("*** Example ***")
            print("example.context: %s " % (example.context))
            print("context_tokens_list: %s " % (context_tokens_list))
            print("context_attention_mask: %s " % (context_attention_mask))
            print("context_token_type_ids: %s " % (context_token_type_ids))

            print("example.response: %s " % (example.response))
            print("response_tokens: %s " % (response_tokens))
            print("response_attention_mask: %s " % (response_attention_mask))
            print("response_token_type_ids: %s " % (response_token_type_ids))

            print("example.personas: %s " % (example.personas))
            print("personas_tokens_list: %s " % (personas_tokens_list))
            print("personas_attention_mask_list: %s " % (personas_attention_mask_list))
            print("personas_token_type_ids_list: %s " % (personas_token_type_ids_list))

            print("label: %s (id = %d)" % (example.label, label))

            if not evaluate:
                print("example.uncorrelated_personas: %s " % (example.uncorrelated_personas))
                print("uncorrelated_personas_tokens_list: %s " % (uncorrelated_personas_tokens_list))
                print("uncorrelated_personas_attention_mask_list: %s " % (uncorrelated_personas_attention_mask_list))
                print("uncorrelated_personas_token_type_ids_list: %s " % (uncorrelated_personas_token_type_ids_list))
                print("uncorrelated_personas_index: %s " % (uncorrelated_personas_index))

                print("example.neg_personas: %s " % (example.neg_personas))
                print("neg_personas_tokens_list: %s " % (neg_personas_tokens_list))
                print("neg_personas_attention_mask_list: %s " % (neg_personas_attention_mask_list))
                print("neg_personas_token_type_ids_list: %s " % (neg_personas_token_type_ids_list))

        if not evaluate:
            features.append(InputFeatures_Multi_Task(
                context_input_ids=context_tokens_list,
                context_attention_mask=context_attention_mask,
                context_token_type_ids=context_token_type_ids,

                response_input_ids=response_tokens,
                response_attention_mask=response_attention_mask,
                response_token_type_ids=response_token_type_ids,

                personas_input_ids=personas_tokens_list,  # [max_persona_num, persona_max_length]
                personas_attention_mask=personas_attention_mask_list,
                personas_token_type_ids=personas_token_type_ids_list,
                label=label,

                uncorrelated_personas_input_ids=uncorrelated_personas_tokens_list,
                # [max_persona_num, persona_max_length]
                uncorrelated_personas_attention_mask=uncorrelated_personas_attention_mask_list,
                uncorrelated_personas_token_type_ids=uncorrelated_personas_token_type_ids_list,
                uncorrelated_personas_index=uncorrelated_personas_index,

                neg_personas_input_ids=neg_personas_tokens_list,  # [max_persona_num, persona_max_length]
                neg_personas_attention_mask=neg_personas_attention_mask_list,
                neg_personas_token_type_ids=neg_personas_token_type_ids_list,
            ))
        else:
            features.append(InputFeatures_Multi_Task(
                context_input_ids=context_tokens_list,
                context_attention_mask=context_attention_mask,
                context_token_type_ids=context_token_type_ids,

                response_input_ids=response_tokens,
                response_attention_mask=response_attention_mask,
                response_token_type_ids=response_token_type_ids,

                personas_input_ids=personas_tokens_list,  # [max_persona_num, persona_max_length]
                personas_attention_mask=personas_attention_mask_list,
                personas_token_type_ids=personas_token_type_ids_list,
                label=label))

    return features


def create_personas_data(args, personas, tokenizer):
    personas_tokens_list = []
    personas_attention_mask_list = []
    personas_token_type_ids_list = []
    for one_p in personas:
        t_one_p = tokenizer.tokenize(one_p)
        if len(t_one_p) > args.persona_max_length - 2:
            t_one_p = t_one_p[:args.persona_max_length - 2]
        t_one_p = ["[CLS]"] + t_one_p + ["[SEP]"]
        t_one_p = tokenizer.convert_tokens_to_ids(t_one_p)

        one_atten_mask = [1] * len(t_one_p)
        if len(t_one_p) < args.persona_max_length:
            padding_length = args.persona_max_length - len(t_one_p)
            t_one_p = t_one_p + [0] * padding_length
            one_atten_mask = one_atten_mask + [0] * padding_length
        one_token_type_ids = [0] * args.persona_max_length

        assert len(t_one_p) == args.persona_max_length
        assert len(one_atten_mask) == args.persona_max_length
        assert len(one_token_type_ids) == args.persona_max_length

        personas_tokens_list.append(t_one_p)
        personas_attention_mask_list.append(one_atten_mask)
        personas_token_type_ids_list.append(one_token_type_ids)

    if len(personas_tokens_list) < args.max_persona_num:
        for i in range(args.max_persona_num - len(personas_tokens_list)):
            personas_tokens_list.append([0] * args.persona_max_length)
            personas_attention_mask_list.append([0] * args.persona_max_length)
            personas_token_type_ids_list.append([0] * args.persona_max_length)

    assert len(personas_tokens_list) == args.max_persona_num and \
           len(personas_tokens_list[0]) == args.persona_max_length
    assert len(personas_attention_mask_list) == args.max_persona_num and \
           len(personas_attention_mask_list[0]) == args.persona_max_length
    assert len(personas_token_type_ids_list) == args.max_persona_num and \
           len(personas_token_type_ids_list[0]) == args.persona_max_length

    return personas_tokens_list, personas_attention_mask_list, personas_token_type_ids_list
