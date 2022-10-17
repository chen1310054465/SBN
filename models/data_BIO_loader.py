
import torch
import numpy as np
import random
import json
from transformers import BertTokenizer

validity2id = {'none': 0, 'positive': 1, 'negative': 1, 'neutral': 1}
sentiment2id = {'none': 0, 'positive': 1, 'negative': 2, 'neutral': 3}


def get_spans(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_subject_labels(tags):
    '''for BIO tag'''

    label = {}
    subject_span = get_spans(tags)[0]
    tags = tags.strip().split()
    sentence = []
    for tag in tags:
        sentence.append(tag.strip().split('\\')[0])
    word = ' '.join(sentence[subject_span[0]:subject_span[1] + 1])
    label[word] = subject_span
    return label


def get_object_labels(tags):
    '''for BIO tag'''
    label = {}
    object_spans = get_spans(tags)
    tags = tags.strip().split()
    sentence = []
    for tag in tags:
        sentence.append(tag.strip().split('\\')[0])
    for object_span in object_spans:
        word = ' '.join(sentence[object_span[0]:object_span[1] + 1])
        label[word] = object_span
    return label


class InputExample(object):
    def __init__(self, id, text_a, aspect_num, triple_num, all_label=None, text_b=None):
        """Build a InputExample"""
        self.id = id
        self.text_a = text_a
        self.text_b = text_b
        self.all_label = all_label
        self.aspect_num = aspect_num
        self.triple_num = triple_num


class Instance(object):
    def __init__(self, sentence_pack, args):
        triple_dict = {}
        id = sentence_pack['id']
        aspect_num = 0
        for triple in sentence_pack['triples']:
            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            sentiment = triple['sentiment']
            subject_label = get_subject_labels(aspect)
            object_label = get_object_labels(opinion)
            objects = list(object_label.keys())
            subject = list(subject_label.keys())[0]
            aspect_num += len(subject_label)
            for i, object in enumerate(objects):
                # 由于数据集的每个triples中aspect只有一个，而opinion可能有多个  需要分开构建
                word = str(subject) + '|' + str(object)
                if word not in triple_dict:
                    triple_dict[word] = []
                triple_dict[word] = (subject_label[subject], object_label[object], sentiment)
        examples = InputExample(id=id, text_a=sentence_pack['sentence'], text_b=None, all_label=triple_dict,
                                aspect_num=aspect_num, triple_num=len(triple_dict))
        self.examples = examples
        self.triple_num = len(triple_dict)
        self.aspect_num = aspect_num


def load_data_instances(sentence_packs, args):
    instances = list()
    triples_num = 0
    aspects_num = 0
    for i, sentence_pack in enumerate(sentence_packs):
        instance = Instance(sentence_pack, args)
        instances.append(instance.examples)
        triples_num += instance.triple_num
        aspects_num += instance.aspect_num
    return instances


def convert_examples_to_features(args, train_instances, max_span_length=8):

    features = []
    num_aspect = 0
    num_triple = 0
    num_opinion = 0
    differ_opinion_senitment_num = 0
    for ex_index, example in enumerate(train_instances):
        sample = {'id': example.id}
        sample['tokens'] = example.text_a.split(' ')
        sample['text_length'] = len(sample['tokens'])
        sample['triples'] = example.all_label
        sample['sentence'] = example.text_a
        aspect = {}
        opinion = {}

        opinion_reverse = {}
        aspect_reverse  = {}

        differ_opinion_sentiment = False

        for triple_name in sample['triples']:
            aspect_span, opinion_span, sentiment = tuple(sample['triples'][triple_name][0]), tuple(
                sample['triples'][triple_name][1]), sample['triples'][triple_name][2]
            num_triple += 1
            if aspect_span not in aspect:
                aspect[aspect_span] = sentiment
                opinion[aspect_span] = [(opinion_span, sentiment)]
            else:
                assert aspect[aspect_span] == sentiment
                opinion[aspect_span].append((opinion_span, sentiment))

            if opinion_span not in opinion_reverse:
                opinion_reverse[opinion_span] = sentiment
                aspect_reverse[opinion_span] = [(aspect_span, sentiment)]
            else:
                '''同一aspect的不同的opinion拥有相同极性，但是'''
                if opinion_reverse[opinion_span] != sentiment:
                    differ_opinion_sentiment = True
                else:
                    aspect_reverse[opinion_span].append((aspect_span, sentiment))
        if differ_opinion_sentiment:
            differ_opinion_senitment_num += 1
            print(ex_index, '单意见词多极性')
            continue

        num_aspect += len(aspect)
        num_opinion += len(opinion)

        # if len(aspect) != example.aspect_num:
        #     print('有不同三元组使用重复了aspect:', example.id)

        spans = []
        span_tokens = []

        spans_aspect_label = []
        spans_aspect2opinion_label =[]
        spans_opinion_label = []

        reverse_opinion_label = []
        reverse_opinion2aspect_label = []
        reverse_aspect_label = []

        if args.order_input:
            for i in range(max_span_length):
                if sample['text_length'] < i:
                    continue
                for j in range(sample['text_length'] - i):
                    spans.append((j, i + j, i + 1))
                    span_token = ' '.join(sample['tokens'][j:i + j + 1])
                    span_tokens.append(span_token)
                    if (j, i + j) not in aspect:
                        spans_aspect_label.append(0)
                    else:
                        # spans_aspect_label.append(sentiment2id[aspect[(j, i + j)]])
                        spans_aspect_label.append(validity2id[aspect[(j, i + j)]])
                    if (j, i + j) not in opinion_reverse:
                        reverse_opinion_label.append(0)
                    else:
                        # reverse_opinion_label.append(sentiment2id[opinion_reverse[(j, i + j)]])
                        reverse_opinion_label.append(validity2id[opinion_reverse[(j, i + j)]])

        else:
            for i in range(sample['text_length']):
                for j in range(i, min(sample['text_length'], i + max_span_length)):
                    spans.append((i, j, j - i + 1))
                    # sample['spans'].append((i, j, j-i+1))
                    span_token = ' '.join(sample['tokens'][i:j + 1])
                    # sample['span tokens'].append(span_tokens)
                    span_tokens.append(span_token)
                    if (i, j) not in aspect:
                        spans_aspect_label.append(0)
                    else:
                        # spans_aspect_label.append(sentiment2id[aspect[(i, j)]])
                        spans_aspect_label.append(validity2id[aspect[(i, j)]])
                    if (i, j) not in opinion_reverse:
                        reverse_opinion_label.append(0)
                    else:
                        # reverse_opinion_label.append(sentiment2id[opinion_reverse[(i, j)]])
                        reverse_opinion_label.append(validity2id[opinion_reverse[(i, j)]])


        assert len(span_tokens) == len(spans)
        for key_aspect in opinion:
            opinion_list = []
            sentiment_opinion = []
            spans_aspect2opinion_label.append(key_aspect)
            for opinion_span_2_aspect in opinion[key_aspect]:
                opinion_list.append(opinion_span_2_aspect[0])
                sentiment_opinion.append(opinion_span_2_aspect[1])
            assert len(set(sentiment_opinion)) == 1
            opinion_label2triple = []
            for i in spans:
                if (i[0], i[1]) not in opinion_list:
                    opinion_label2triple.append(0)
                else:
                    opinion_label2triple.append(sentiment2id[sentiment_opinion[0]])
            spans_opinion_label.append(opinion_label2triple)

        for opinion_key in aspect_reverse:
            aspect_list = []
            sentiment_aspect = []
            reverse_opinion2aspect_label.append(opinion_key)
            for aspect_span_2_opinion in aspect_reverse[opinion_key]:
                aspect_list.append(aspect_span_2_opinion[0])
                sentiment_aspect.append(aspect_span_2_opinion[1])
            assert len(set(sentiment_aspect)) == 1
            aspect_label2triple = []
            for i in spans:
                if (i[0], i[1]) not in aspect_list:
                    aspect_label2triple.append(0)
                else:
                    aspect_label2triple.append(sentiment2id[sentiment_aspect[0]])
            reverse_aspect_label.append(aspect_label2triple)

        sample['aspect_num'] = len(spans_opinion_label)
        sample['spans_aspect2opinion_label'] = spans_aspect2opinion_label
        sample['reverse_opinion_num'] = len(reverse_aspect_label)
        sample['reverse_opinion2aspect_label'] = reverse_opinion2aspect_label

        if args.random_shuffle != 0:
            np.random.seed(args.random_shuffle)
            shuffle_ix = np.random.permutation(np.arange(len(spans)))
            spans_np = np.array(spans)[shuffle_ix]
            span_tokens_np = np.array(span_tokens)[shuffle_ix]
            '''双向同顺序打乱'''
            spans_aspect_label_np = np.array(spans_aspect_label)[shuffle_ix]
            reverse_opinion_label_np = np.array(reverse_opinion_label)[shuffle_ix]
            spans_opinion_label_shuffle = []
            for spans_opinion_label_split in spans_opinion_label:
                spans_opinion_label_split_np = np.array(spans_opinion_label_split)[shuffle_ix]
                spans_opinion_label_shuffle.append(spans_opinion_label_split_np.tolist())
            spans_opinion_label = spans_opinion_label_shuffle
            reverse_aspect_label_shuffle = []
            for reverse_aspect_label_split in reverse_aspect_label:
                reverse_aspect_label_split_np = np.array(reverse_aspect_label_split)[shuffle_ix]
                reverse_aspect_label_shuffle.append(reverse_aspect_label_split_np.tolist())
            reverse_aspect_label = reverse_aspect_label_shuffle
            spans, span_tokens, spans_aspect_label, reverse_opinion_label  = spans_np.tolist(), span_tokens_np.tolist(),\
                                                                             spans_aspect_label_np.tolist(), reverse_opinion_label_np.tolist()
        related_spans = np.zeros((len(spans), len(spans)), dtype=int)
        for i in range(len(span_tokens)):
            span_token = span_tokens[i].split(' ')
            # for j in range(i, len(span_tokens)):
            for j in range(len(span_tokens)):
                differ_span_token = span_tokens[j].split(' ')
                if set(span_token) & set(differ_span_token) == set():
                    related_spans[i, j] = 0
                else:
                    related_spans[i, j] = 1

        sample['related_span_array'] = related_spans
        sample['spans'], sample['span tokens'], sample['spans_aspect_label'], sample[
            'spans_opinion_label'] = spans, span_tokens, spans_aspect_label, spans_opinion_label
        sample['reverse_opinion_label'], sample['reverse_aspect_label'] = reverse_opinion_label, reverse_aspect_label
        features.append(sample)
    return features, num_aspect, num_opinion


def load_data(args, path, if_train=False):
    # sentence_packs = json.load(open(path))
    # if if_train:
    #     random.seed(args.RANDOM_SEED)
    #     random.shuffle(sentence_packs)
    # instances = load_data_instances(sentence_packs, args)
    # tokenizer = BertTokenizer.from_pretrained(args.init_vocab, do_lower_case=args.do_lower_case)
    # data_instances, aspect_num, num_opinion = convert_examples_to_features(args, train_instances=instances,
    #                                                                        max_seq_length=args.max_seq_length,
    #                                                                        tokenizer=tokenizer,
    #                                                                        max_span_length=args.max_span_length)
    # list_instance_batch = []
    # for i in range(0, len(data_instances), args.train_batch_size):
    #     list_instance_batch.append(data_instances[i:i + args.train_batch_size])
    # return list_instance_batch

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if if_train:
        random.seed(args.RANDOM_SEED)
        random.shuffle(lines)
    instances = load_data_instances_txt(lines)
    data_instances, aspect_num, num_opinion = convert_examples_to_features(args, train_instances=instances,
                                                                           max_span_length=args.max_span_length)
    list_instance_batch = []
    for i in range(0, len(data_instances), args.train_batch_size):
        list_instance_batch.append(data_instances[i:i + args.train_batch_size])
    return list_instance_batch


def load_data_instances_txt(lines):
    sentiment2sentiment = {'NEG': 'negative', 'POS': 'positive', 'NEU': 'neutral'}

    instances = list()
    triples_num = 0
    aspects_num = 0
    for ex_index, line in enumerate(lines):
        id = str(ex_index)  # id
        line = line.strip()
        line = line.split('####')
        sentence = line[0].split()  # sentence
        raw_pairs = eval(line[1])  # triplets

        triple_dict = {}
        aspect_num = 0
        for triple in raw_pairs:
            raw_aspect = triple[0]
            raw_opinion = triple[1]
            sentiment = sentiment2sentiment[triple[2]]

            if len(raw_aspect) == 1:
                aspect_word = sentence[raw_aspect[0]]
                raw_aspect = [raw_aspect[0], raw_aspect[0]]
            else:
                aspect_word = ' '.join(sentence[raw_aspect[0]: raw_aspect[-1] + 1])
            aspect_label = {}
            aspect_label[aspect_word] = [raw_aspect[0], raw_aspect[-1]]
            aspect_num += len(aspect_label)

            if len(raw_opinion) == 1:
                opinion_word = sentence[raw_opinion[0]]
                raw_opinion = [raw_opinion[0], raw_opinion[0]]
            else:
                opinion_word = ' '.join(sentence[raw_opinion[0]: raw_opinion[-1] + 1])
            opinion_label = {}
            opinion_label[opinion_word] = [raw_opinion[0], raw_opinion[-1]]

            word = str(aspect_word) + '|' + str(opinion_word)
            if word not in triple_dict:
                triple_dict[word] = []
                triple_dict[word] = ([raw_aspect[0], raw_aspect[-1]], [raw_opinion[0], raw_opinion[-1]], sentiment)
            else:
                print('单句' + id + '中三元组重复出现！')
        examples = InputExample(id=id, text_a=line[0], text_b=None, all_label=triple_dict, aspect_num=aspect_num,
                                triple_num=len(triple_dict))

        instances.append(examples)
        triples_num += triples_num
        aspects_num += aspect_num

    return instances


class DataTterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = len(instances)
        self.tokenizer = BertTokenizer.from_pretrained(args.init_vocab, do_lower_case=args.do_lower_case)

    def get_batch(self, batch_num):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        spans_aspect_tensor_list = []
        spans_opinion_label_tensor_list = []

        reverse_ner_label_tensor_list = []
        reverse_opinion_tensor_list = []
        reverse_aspect_tensor_list = []
        sentence_length = []
        related_spans_list = []

        max_tokens = self.args.max_seq_length
        max_spans = 0
        for i, sample in enumerate(self.instances[batch_num]):
            tokens = sample['tokens']
            spans = sample['spans']
            span_tokens = sample['span tokens']
            spans_ner_label = sample['spans_aspect_label']
            spans_aspect2opinion_labels = sample['spans_aspect2opinion_label']
            spans_opinion_label = sample['spans_opinion_label']

            reverse_ner_label = sample['reverse_opinion_label']
            reverse_opinion2aspect_labels = sample['reverse_opinion2aspect_label']
            reverse_aspect_label = sample['reverse_aspect_label']

            related_spans = sample['related_span_array']
            spans_aspect_labels, reverse_opinion_labels = [], []
            for spans_aspect2opinion_label in spans_aspect2opinion_labels:
                spans_aspect_labels.append((i, spans_aspect2opinion_label[0], spans_aspect2opinion_label[1]))
            for reverse_opinion2aspect_label in reverse_opinion2aspect_labels:
                reverse_opinion_labels.append((i, reverse_opinion2aspect_label[0], reverse_opinion2aspect_label[1]))
            bert_tokens, tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_aspect_labels_tensor, spans_opinion_tensor, \
            reverse_ner_label_tensor, reverse_opinion_tensor, reverse_aspect_tensor = \
                self.get_input_tensors(self.tokenizer, tokens, spans, spans_ner_label, spans_aspect_labels,
                                         spans_opinion_label, reverse_ner_label, reverse_opinion_labels, reverse_aspect_label)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            spans_aspect_tensor_list.append(spans_aspect_labels_tensor)
            spans_opinion_label_tensor_list.append(spans_opinion_tensor)
            reverse_ner_label_tensor_list.append(reverse_ner_label_tensor)
            reverse_opinion_tensor_list.append(reverse_opinion_tensor)
            reverse_aspect_tensor_list.append(reverse_aspect_tensor)
            assert bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1] == reverse_ner_label_tensor.shape[1]
            # tokens和spans的最大个数被设定为固定值
            # if (tokens_tensor.shape[1] > max_tokens):
            #     max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append((bert_tokens, tokens_tensor.shape[1], bert_spans_tensor.shape[1]))
            related_spans_list.append(related_spans)
        '''由于不同句子方阵不一样大，所以先不转为tensor'''
        #related_spans_tensor = torch.tensor(related_spans_list)
        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_spans_mask_tensor = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_aspect_tensor = None
        final_spans_opinion_label_tensor = None

        final_reverse_ner_label_tensor = None
        final_reverse_opinion_tensor = None
        final_reverse_aspect_label_tensor = None
        final_related_spans_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_aspect_tensor, spans_opinion_label_tensor, \
            reverse_ner_label_tensor, reverse_opinion_tensor, reverse_aspect_tensor, related_spans \
                in zip(tokens_tensor_list, bert_spans_tensor_list, spans_ner_label_tensor_list, spans_aspect_tensor_list,
                       spans_opinion_label_tensor_list, reverse_ner_label_tensor_list, reverse_opinion_tensor_list,
                       reverse_aspect_tensor_list, related_spans_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
            if tokens_pad_length > 0:
                pad = torch.full([1, tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1, tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            num_aspect = spans_aspect_tensor.shape[1]
            num_opinion = reverse_opinion_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full([1, spans_pad_length, bert_spans_tensor.shape[2]], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)

                mask_pad = torch.full([1, spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)
                reverse_ner_label_tensor = torch.cat((reverse_ner_label_tensor, mask_pad), dim=1)

                opinion_mask_pad = torch.full([1, num_aspect, spans_pad_length], 0, dtype=torch.long)
                spans_opinion_label_tensor = torch.cat((spans_opinion_label_tensor, opinion_mask_pad), dim=-1)
                aspect_mask_pad = torch.full([1, num_opinion, spans_pad_length], 0, dtype=torch.long)
                reverse_aspect_tensor = torch.cat((reverse_aspect_tensor, aspect_mask_pad), dim=-1)
                '''对span类似方阵mask'''
                related_spans_pad_1 = np.zeros([num_spans, spans_pad_length])
                related_spans_pad_2 = np.zeros([spans_pad_length, max_spans])
                related_spans_hstack = np.hstack((related_spans, related_spans_pad_1))
                related_spans = np.vstack((related_spans_hstack, related_spans_pad_2))
            related_spans_tensor = torch.as_tensor(torch.from_numpy(related_spans), dtype=torch.bool)
            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor

                final_bert_spans_tensor = bert_spans_tensor
                final_spans_mask_tensor = spans_mask_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_aspect_tensor = spans_aspect_tensor.squeeze(0)
                final_spans_opinion_label_tensor = spans_opinion_label_tensor.squeeze(0)
                final_reverse_ner_label_tensor = reverse_ner_label_tensor
                final_reverse_opinion_tensor = reverse_opinion_tensor.squeeze(0)
                final_reverse_aspect_label_tensor = reverse_aspect_tensor.squeeze(0)
                final_related_spans_tensor = related_spans_tensor.unsqueeze(0)
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)

                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_spans_aspect_tensor = torch.cat(
                    (final_spans_aspect_tensor, spans_aspect_tensor.squeeze(0)), dim=0)
                final_spans_opinion_label_tensor = torch.cat(
                    (final_spans_opinion_label_tensor, spans_opinion_label_tensor.squeeze(0)), dim=0)
                final_reverse_ner_label_tensor = torch.cat(
                    (final_reverse_ner_label_tensor, reverse_ner_label_tensor), dim=0)
                final_reverse_opinion_tensor = torch.cat(
                    (final_reverse_opinion_tensor,  reverse_opinion_tensor.squeeze(0)), dim=0)
                final_reverse_aspect_label_tensor = torch.cat(
                    (final_reverse_aspect_label_tensor, reverse_aspect_tensor.squeeze(0)), dim=0)
                final_related_spans_tensor = torch.cat(
                    (final_related_spans_tensor, related_spans_tensor.unsqueeze(0)), dim=0)

        # 注意，特征中最大span间隔不一定为设置的max_span_length，这是因为bert分词之后造成的span扩大了。
        final_tokens_tensor = final_tokens_tensor.to(self.args.device)
        final_attention_mask = final_attention_mask.to(self.args.device)
        final_bert_spans_tensor = final_bert_spans_tensor.to(self.args.device)
        final_spans_mask_tensor = final_spans_mask_tensor.to(self.args.device)
        final_spans_ner_label_tensor = final_spans_ner_label_tensor.to(self.args.device)
        final_spans_aspect_tensor = final_spans_aspect_tensor.to(self.args.device)
        final_spans_opinion_label_tensor = final_spans_opinion_label_tensor.to(self.args.device)
        final_reverse_ner_label_tensor = final_reverse_ner_label_tensor.to(self.args.device)
        final_reverse_opinion_tensor = final_reverse_opinion_tensor.to(self.args.device)
        final_reverse_aspect_label_tensor = final_reverse_aspect_label_tensor.to(self.args.device)
        final_related_spans_tensor = final_related_spans_tensor.to(self.args.device)
        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, \
               final_spans_ner_label_tensor, final_spans_aspect_tensor, final_spans_opinion_label_tensor, \
               final_reverse_ner_label_tensor, final_reverse_opinion_tensor, final_reverse_aspect_label_tensor, \
               final_related_spans_tensor, sentence_length


    def get_input_tensors(self, tokenizer, tokens, spans, spans_ner_label, spans_aspect_label, spans_opinion_label,
                          reverse_ner_label, reverse_opinion_labels, reverse_aspect_label):
        start2idx = []
        end2idx = []
        bert_tokens = []
        bert_tokens.append(tokenizer.cls_token)
        for token in tokens:
            start2idx.append(len(bert_tokens))
            test_1 = len(bert_tokens)
            sub_tokens = tokenizer.tokenize(token)
            if self.args.span_generation == "CNN":
                bert_tokens.append(sub_tokens[0])
            elif self.args.Only_token_head:
                bert_tokens.append(sub_tokens[0])
            else:
                bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens) - 1)
            test_2 = len(bert_tokens) - 1
            # if test_2 != test_1:
            #     print("差异：", test_2 - test_1)
            # else:
            #     print("no extra token")
        bert_tokens.append(tokenizer.sep_token)
        indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        # 在bert分出subword之后  需要对原有的aspect span进行补充
        spans_aspect_label = [[aspect_span[0], start2idx[aspect_span[1]], end2idx[aspect_span[2]]] for
                              aspect_span in spans_aspect_label]
        reverse_opinion_label =[[opinion_span[0], start2idx[opinion_span[1]], end2idx[opinion_span[2]]] for
                                opinion_span in reverse_opinion_labels]
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])
        spans_aspect_tensor = torch.tensor([spans_aspect_label])
        spans_opinion_tensor = torch.tensor([spans_opinion_label])
        reverse_ner_label_tensor = torch.tensor([reverse_ner_label])
        reverse_opinion_tensor = torch.tensor([reverse_opinion_label])
        reverse_aspect_tensor = torch.tensor([reverse_aspect_label])
        return bert_tokens,tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_aspect_tensor, spans_opinion_tensor, \
               reverse_ner_label_tensor, reverse_opinion_tensor, reverse_aspect_tensor
