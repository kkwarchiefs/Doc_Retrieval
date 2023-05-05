# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import pickle
import random
from typing import Union, List, Tuple, Dict

import datasets
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
import json
from .arguments import DataArguments, RerankerTrainingArguments

class GroupedTrainDuQA(Dataset):
    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args
        # with open(args.passage_path + '/passage2id.map.json', "r") as fr:
        #     self.pcid2pid = json.load(fr)
        # self.idx2txt = self.read_part(args.passage_path)
        self.idx2txt = pickle.load(open(args.passage_path, 'rb'))

    def read_part(self, passage_path):
        qid2txt = []
        for i in range(4):
            for line in open(passage_path + '/part-0' + str(i), 'r', encoding='utf-8'):
                items = line.strip().split('\t')
                qid2txt.append(items[2])
        return qid2txt

    def __len__(self):
        return self.total_len

    def create_one_example(self, doc_encoding: str):
        item = self.tok.encode_plus(
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qtext = group['qry']
        pos_pid = random.choice(group['pos'])
        neg_group = group['neg']
        if len(neg_group) < self.args.train_group_size:
            negs = random.choices(neg_group, k=self.args.train_group_size)
        else:
            negs = random.sample(neg_group, k=self.args.train_group_size)
        negs[0] = pos_pid
        group_batch = []
        for neg_id in negs:
            psg = '问题：' + qtext + '，答案：' + self.idx2txt[int(neg_id)]
            group_batch.append(self.create_one_example(psg))
        return group_batch

class PredictionDuQA(Dataset):
    columns = [
        'qid', 'pid', 'qry', 'psg'
    ]

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.max_len = max_len
        self.args = args
        # self.idx2txt = self.read_part(args.passage_path)
        self.idx2txt = pickle.load(open(args.passage_path, 'rb'))

    def read_part(self, passage_path):
        qid2txt = []
        for i in range(4):
            for line in open(passage_path + '/part-0' + str(i), 'r', encoding='utf-8'):
                items = line.strip().split('\t')
                qid2txt.append(items[2])
        return qid2txt

    def __len__(self):
        return len(self.nlp_dataset)

    def create_one_example(self, doc_encoding: str):
        item = self.tok.encode_plus(
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]['text'].split('\t')
        qtext = group[0]
        neg_id = group[1]
        psg = '问题：' + qtext + '，答案：' + self.idx2txt[int(neg_id)]
        return self.create_one_example(psg)
        # qtext = group['qry']
        # neg_group = group['neg']
        # group_batch = []
        # for neg_id in neg_group:
        #     psg = '问题：' + qtext + '，答案：' + self.idx2txt[int(neg_id)]
        #     group_batch.append(self.create_one_example(psg))
        # return group_batch


class GroupedTrainSquad(Dataset):
    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_tsv,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args

    def __len__(self):
        return self.total_len

    def create_one_example(self, doc_encoding: str):
        item = self.tok.encode_plus(
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]['text'].split('\t')
        qtext = group[0]
        pos_pid = group[1]
        neg_group = [group[1], group[2]]
        while len(neg_group) < self.args.train_group_size:
            idx = random.randint(0, self.__len__()-1)
            neg_case = self.nlp_dataset[idx]['text'].split('\t')
            if neg_case[1] != pos_pid:
                neg_group.append(neg_case[2])
        group_batch = []
        for neg_text in neg_group:
            psg = 'Q:' + qtext + 'A:' + neg_text
            group_batch.append(self.create_one_example(psg))
        return group_batch

class PredictionSquad(Dataset):

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.max_len = max_len
        self.args = args

    def __len__(self):
        return len(self.nlp_dataset)

    def create_one_example(self, doc_encoding: str):
        item = self.tok.encode_plus(
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]['text'].split('\t')
        qtext = group[0]
        neg_text = group[1]
        psg = 'Q:' + qtext + 'A:' + neg_text
        return self.create_one_example(psg)
        # qtext = group['qry']
        # neg_group = group['neg']
        # group_batch = []
        # for neg_id in neg_group:
        #     psg = '问题：' + qtext + '，答案：' + self.idx2txt[int(neg_id)]
        #     group_batch.append(self.create_one_example(psg))
        # return group_batch

class GroupedTrainMSMul(Dataset):
    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args
        root = '/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/'
        self.idx2zh = pickle.load(open(root + "passage_idx.pkl", 'rb'))
        self.idx2en = pickle.load(open(root + "en_passage_idx.pkl", 'rb'))
        self.zhlist = list(range(0, len(self.idx2zh)-10))
        self.enlist = list(range(0, len(self.idx2en)-10))

    def __len__(self):
        return self.total_len

    def create_one_example(self, doc_encoding: str):
        item = self.tok.encode_plus(
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        is_english = group['zh'] != group['qry']
        if random.randint(0, 1) == 0:
            qtext = group['zh']
        else:
            qtext = group['en']
            if qtext == "NULL":
                qtext = group['zh']
        pos_pid = random.choice(group['pos'])
        neg_group = group['neg']
        hard_neg = int(self.args.train_group_size / 3 * 2)
        if len(neg_group) < hard_neg:
            negs = random.choices(neg_group, k=hard_neg)
        else:
            negs = random.sample(neg_group, k=hard_neg)
        rand_neg = self.args.train_group_size - hard_neg
        if is_english:
            ids = random.sample(self.zhlist, k=rand_neg)
        else:
            ids = random.sample(self.enlist, k=rand_neg)
        negs += ids
        negs[0] = pos_pid
        group_batch = []
        for neg_id in negs:
            if is_english:
                psg = self.idx2en[int(neg_id)]
            else:
                psg = self.idx2zh[int(neg_id)]
            qry_psg = 'Q:' + qtext + 'A:' + psg
            group_batch.append(self.create_one_example(qry_psg))
        return group_batch

class PredictionMSMul(Dataset):

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.max_len = max_len
        self.args = args

    def __len__(self):
        return len(self.nlp_dataset)

    def create_one_example(self, doc_encoding: str):
        item = self.tok.encode_plus(
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]['text'].split('\t')
        qtext = group[0]
        ptext = group[1]
        psg = 'Q:' + qtext + 'A:' + ptext
        return self.create_one_example(psg)

class GroupedTrainDatasetURLTitle(Dataset):
    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args
        query_file = args.corpus_path + 'queries.all.tsv'
        collection_file = args.corpus_path + 'collection.tsv'
        title_file = args.corpus_path + 'para.title.txt'
        self.qid2txt = self.read_txt(query_file)
        self.pid2txt = self.read_txt(collection_file)
        self.pid2title = self.read_txt(title_file)

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def __len__(self):
        return self.total_len

    def create_one_example(self, doc_encoding: str):
        item = self.tok.encode_plus(
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qid = group['qry']
        qry = self.qid2txt[qid]
        pos_pid = random.choice(group['pos'])
        neg_group = group['neg']
        if len(neg_group) < self.args.train_group_size:
            negs = random.choices(neg_group, k=self.args.train_group_size)
        else:
            negs = random.sample(neg_group, k=self.args.train_group_size)
        negs[0] = pos_pid
        group_batch = []
        for neg_id in negs:
            title = self.pid2title.get(neg_id)
            if title == '-':
                title = 'null'
            psg = qry + ', title:' + title + ', text: ' + self.pid2txt[neg_id]
            group_batch.append(self.create_one_example(psg))
        return group_batch


class GroupedTrainDatasetURLTitleGlobal(Dataset):
    query_columns = ['qid', 'query']
    document_columns = ['pid', 'passage']

    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args
        query_file = args.corpus_path + 'queries.all.tsv'
        collection_file = args.corpus_path + 'collection.tsv'
        title_file = args.corpus_path + 'para.title.txt'
        self.qid2txt = self.read_txt(query_file)
        self.pid2txt = self.read_txt(collection_file)
        self.pid2title = self.read_txt(title_file)

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def __len__(self):
        return self.total_len

    def create_one_example(self, doc_encoding: str):
        item = self.tok.encode_plus(
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qid = group['qry']
        qry = self.qid2txt[qid]
        pos_pid = random.choice(group['pos'])
        neg_group = group['neg']  # [nid for nid in group['neg'] if nid not in adds]
        if len(neg_group) < self.args.train_group_size:
            negs = random.choices(neg_group, k=self.args.train_group_size)
        else:
            negs = random.sample(neg_group, k=self.args.train_group_size)
        idx = random.randint(0, self.args.train_group_size - 1)
        negs[idx] = pos_pid
        group_batch = []
        for neg_id in negs:
            title = self.pid2title.get(neg_id)
            if title == '-':
                title = 'null'
            psg = qry + ', title:' + title + ', text: ' + self.pid2txt[neg_id]
            item = self.create_one_example(psg)
            item['label'] = idx
            group_batch.append(item)
        return group_batch


class GroupedTrainDatasetClassifyLongQry(Dataset):
    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args
        query_file = args.corpus_path + 'queries.all.tsv'
        collection_file = args.corpus_path + 'collection.tsv'
        title_file = args.corpus_path + 'para.title.txt'
        self.qid2txt = self.read_txt(query_file)
        self.pid2txt = self.read_txt(collection_file)
        self.pid2title = self.read_txt(title_file)

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qid = group['qry']
        qry = self.qid2txt[qid]
        pos_pid = random.choice(group['pos'])
        neg_group = group['neg']  # [nid for nid in group['neg'] if nid not in adds]
        negs = neg_group[:self.args.train_group_size]
        random.shuffle(negs)
        label = random.choice(range(self.args.train_group_size))
        negs[label] = pos_pid
        input_tokens = []
        for neg_id in negs:
            # seg.append(len(input_tokens) + 1)
            input_tokens.append(self.tok.sep_token)
            title = self.pid2title.get(neg_id)
            if title == '-':
                title = 'null'
            psg = qry + ', title:' + title + ', text: ' + self.pid2txt[neg_id]
            cd_tk = self.tok.tokenize(psg)[:self.args.max_len]
            input_tokens += cd_tk
        # while len(input_tokens) < 50 * 64:
        #     input_tokens.append(self.tok.pad_token)
        while len(input_tokens) > 0.9 * self.args.train_group_size * self.args.max_len:
            new_input = []
            last = 0
            for idx in range(len(input_tokens) - 1):
                if input_tokens[idx + 1] != self.tok.sep_token:
                    new_input.append(input_tokens[idx])
                else:
                    if idx - last < 100:
                        new_input.append(input_tokens[idx])
                    last = idx
            input_tokens = new_input
        seg = [idx + 1 for idx, a in enumerate(input_tokens) if a == self.tok.sep_token]
        item = self.tok.encode_plus(
            input_tokens,
            truncation=True,
            max_length=int(0.9 * self.args.train_group_size * self.args.max_len) + 100,
            padding=False,
        )
        item['label'] = label
        item['seg'] = seg
        return item


class GroupedTrainDatasetClassifyLongQryRandom(Dataset):
    query_columns = ['qid', 'query']
    document_columns = ['pid', 'passage']

    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args
        query_file = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/queries.all.tsv'
        collection_file = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/collection.tsv'
        title_file = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/para.title.txt'
        self.qid2txt = self.read_txt(query_file)
        self.pid2txt = self.read_txt(collection_file)
        self.pid2title = self.read_txt(title_file)

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qid = group['qry']
        qry = self.qid2txt[qid]
        pos_pid = random.choice(group['pos'])
        neg_group = group['neg']  # [nid for nid in group['neg'] if nid not in adds]
        # if len(neg_group) < self.args.train_group_size:
        #     negs = random.choices(neg_group, k=self.args.train_group_size)
        # else:
        #     negs = random.sample(neg_group, k=self.args.train_group_size)
        random.shuffle(neg_group)
        negs = neg_group[:self.args.train_group_size]
        label = random.choice(range(self.args.train_group_size))
        negs[label] = pos_pid
        # input_tokens = self.tok.tokenize(qry)
        # qry_len = len(input_tokens)
        # seg = []
        input_tokens = []
        for neg_id in negs:
            # seg.append(len(input_tokens) + 1)
            input_tokens.append(self.tok.sep_token)
            title = self.pid2title.get(neg_id)
            if title == '-':
                title = 'null'
            psg = qry + ', title:' + title + ', text: ' + self.pid2txt[neg_id]
            cd_tk = self.tok.tokenize(psg)[:self.args.max_len]
            input_tokens += cd_tk
        # while len(input_tokens) < 50 * 64:
        #     input_tokens.append(self.tok.pad_token)
        while len(input_tokens) > 0.9 * self.args.train_group_size * self.args.max_len:
            new_input = []
            last = 0
            for idx in range(len(input_tokens) - 1):
                if input_tokens[idx + 1] != self.tok.sep_token:
                    new_input.append(input_tokens[idx])
                else:
                    if idx - last < 100:
                        new_input.append(input_tokens[idx])
                    last = idx
            input_tokens = new_input
        seg = [idx + 1 for idx, a in enumerate(input_tokens) if a == self.tok.sep_token]
        item = self.tok.encode_plus(
            input_tokens,
            truncation=True,
            max_length=int(0.9 * self.args.train_group_size * self.args.max_len) + 100,
            padding=False,
        )
        item['label'] = label
        item['seg'] = seg
        return item


class PredictionDatasetURLTitle(Dataset):
    columns = [
        'qid', 'pid', 'qry', 'psg'
    ]

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.max_len = max_len
        query_file = args.corpus_path + 'queries.all.tsv'
        collection_file = args.corpus_path + 'collection.tsv'
        title_file = args.corpus_path + 'para.title.txt'
        self.qid2txt = self.read_txt(query_file)
        self.pid2txt = self.read_txt(collection_file)
        self.pid2title = self.read_txt(title_file)

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def __len__(self):
        return len(self.nlp_dataset)

    def __getitem__(self, item):
        qid, pid, qry, psg = (self.nlp_dataset[item][f] for f in self.columns)
        title = self.pid2title.get(pid)
        qry = self.qid2txt[qid]
        if title == '-':
            title = 'null'
        psg = qry + ', title:' + title + ', text: ' + self.pid2txt[pid]
        return self.tok.encode_plus(
            psg,
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )


class PredictionDatasetClassifyLongQry(Dataset):

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_json,
        )['train']
        self.args = args
        self.tok = tokenizer
        self.max_len = max_len
        query_file = args.corpus_path + 'queries.all.tsv'
        collection_file = args.corpus_path + 'collection.tsv'
        title_file = args.corpus_path + 'para.title.txt'
        self.qid2txt = self.read_txt(query_file)
        self.pid2txt = self.read_txt(collection_file)
        self.pid2title = self.read_txt(title_file)

    def __len__(self):
        return len(self.nlp_dataset)

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qid = group['qry']
        qry = self.qid2txt[qid]
        negs = group['neg'][:self.args.train_group_size]
        input_tokens = []
        for neg_id in negs:
            # seg.append(len(input_tokens) + 1)
            input_tokens.append(self.tok.sep_token)
            title = self.pid2title.get(neg_id)
            if title == '-':
                title = 'null'
            psg = qry + ', title:' + title + ', text: ' + self.pid2txt[neg_id]
            cd_tk = self.tok.tokenize(psg)[:self.args.max_len]
            input_tokens += cd_tk
        # while len(input_tokens) < 11 * 64:
        #     input_tokens.append(self.tok.pad_token)
        while len(input_tokens) > 0.9 * self.args.train_group_size * self.args.max_len:
            new_input = []
            last = 0
            for idx in range(len(input_tokens) - 1):
                if input_tokens[idx + 1] != self.tok.sep_token:
                    new_input.append(input_tokens[idx])
                else:
                    if idx - last < 100:
                        new_input.append(input_tokens[idx])
                    last = idx
            input_tokens = new_input
        seg = [idx + 1 for idx, a in enumerate(input_tokens) if a == self.tok.sep_token]
        item = self.tok.encode_plus(
            input_tokens,
            truncation=True,
            max_length=int(0.9 * self.args.train_group_size * self.args.max_len) + 100,
            padding=False,
        )
        item['label'] = 0
        item['seg'] = seg
        return item


class PredictionDatasetGroupGlobal(Dataset):
    query_columns = ['qid', 'query']
    document_columns = ['pid', 'passage']

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_json,
        )['train']
        self.args = args
        self.tok = tokenizer
        self.max_len = max_len
        query_file = args.corpus_path + 'queries.all.tsv'
        collection_file = args.corpus_path + 'collection.tsv'
        title_file = args.corpus_path + 'para.title.txt'
        self.qid2txt = self.read_txt(query_file)
        self.pid2txt = self.read_txt(collection_file)
        self.pid2title = self.read_txt(title_file)

    def __len__(self):
        return len(self.nlp_dataset)

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def create_one_example(self, doc_encoding: str):
        item = self.tok.encode_plus(
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qid = group['qry']
        qry = self.qid2txt[qid]
        negs = group['neg'][:self.args.train_group_size]
        group_batch = []
        for neg_id in negs:
            title = self.pid2title.get(neg_id)
            if title == '-':
                title = 'null'
            psg = qry + ', title:' + title + ', text: ' + self.pid2txt[neg_id]
            group_batch.append(self.create_one_example(psg))
        return group_batch

class TrainDatasetOpenQAInteract(Dataset):
    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args

    def __len__(self):
        return self.total_len

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        return self.tok.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation='only_second',
            max_length=self.args.max_len,
            padding=False,
        )

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qry = group['qry']
        if len(group['pos']):
            pos = random.choice(group['pos'])
        else:
            pos = qry
        neg_group = [] + group['neg']
        while len(neg_group) < self.args.train_group_size:
            other = self.nlp_dataset[random.choice(range(self.total_len))]
            neg_group += other['neg']
        negs = random.sample(neg_group, k=self.args.train_group_size)
        idx = random.randint(0, self.args.train_group_size - 1)
        negs[idx] = pos
        group_batch = []
        for psg in negs:
            item = self.create_one_example(qry, psg)
            item['label'] = idx
            group_batch.append(item)
        return group_batch

class TrainDatasetOpenQA(Dataset):
    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args

    def __len__(self):
        return self.total_len

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        return self.tok.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation='only_second',
            max_length=self.args.max_len,
            padding=False,
        )

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qry = group['qry']
        if len(group['pos']):
            pos = random.choice(group['pos'])
        else:
            pos = qry
        neg_group = [] + group['neg']
        while len(neg_group) < self.args.train_group_size:
            other = self.nlp_dataset[random.choice(range(self.total_len))]
            neg_group += other['neg']
        negs = random.sample(neg_group, k=self.args.train_group_size)
        negs[0] = pos
        group_batch = []
        for psg in negs:
            item = self.create_one_example(qry, psg)
            group_batch.append(item)
        return group_batch

class PredictionDatasetOpenQA(Dataset):

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.max_len = max_len
        self.args = args
    def __len__(self):
        return len(self.nlp_dataset)

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        item = self.tok.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation='only_second',
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item):
        qry, psg, label = self.nlp_dataset[item]['text'].split('\t')
        return self.tok.encode_plus(
            qry,
            psg,
            truncation='only_second',
            max_length=self.max_len,
            padding=False,
        )


class PredictionDatasetOpenQAInteract(Dataset):

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.max_len = max_len
        self.args = args
    def __len__(self):
        return len(self.nlp_dataset)

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        item = self.tok.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation='only_second',
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item):
        obj = self.nlp_dataset[item]
        qry, psgs,  = obj['qry'], obj['psgs']
        group = []
        for psg in psgs:
            item = self.tok.encode_plus(
                qry,
                psg,
                truncation='only_second',
                max_length=self.max_len,
                padding=False,
            )
            item['mask'] = 1
            group.append(item)
        while len(group) < self.args.eval_group_size:
            item = self.tok.encode_plus(
                qry,
                '',
                truncation='only_second',
                max_length=self.max_len,
                padding=False,
            )
            item['mask'] = 0
            group.append(item)
        return group


class EventTrainDataset(Dataset):
    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args

    def __len__(self):
        return self.total_len

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        item = self.tok.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation='only_second',
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qid = group['qry']
        pos_pid = group['pos'][0]
        neg_group = group['neg'][:self.args.train_group_size-1]
        group_batch = []
        for iid in [pos_pid] + neg_group:
            group_batch.append(self.create_one_example(qid, iid))
        return group_batch

class EventPredictionDataset(Dataset):
    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.args = args
        self.max_len = max_len

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def __len__(self):
        return len(self.nlp_dataset)

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        item = self.tok.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation='only_second',
            max_length=self.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item):
        group = self.nlp_dataset[item]
        qid = group['qry']
        pos_pid = group['pos'][0]
        neg_group = group['neg'][:self.args.train_group_size-1]
        group_batch = []
        for iid in [pos_pid] + neg_group:
            group_batch.append(self.create_one_example(qid, iid))
        return group_batch




@dataclass
class GroupCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)


@dataclass
class ClassifyCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        print(features)
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)
