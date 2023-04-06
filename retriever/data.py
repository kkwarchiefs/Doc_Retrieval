# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
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
import pickle

class RetrivalTrainDataset(Dataset):
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

    def create_one_example(self, text: str, is_query=False):
        item = self.tok.encode_plus(
            text,
            truncation='only_first',
            return_attention_mask=False,
            max_length=self.args.q_max_len if is_query else self.args.p_max_len,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qid = group['qry']
        encoded_query = self.create_one_example(self.qid2txt[qid], is_query=True)

        pos_pid = random.choice(group['pos'])
        neg_group = group['neg']  # [nid for nid in group['neg'] if nid not in adds]
        if len(neg_group) < self.args.train_group_size:
            negs = random.choices(neg_group, k=self.args.train_group_size)
        else:
            negs = random.sample(neg_group, k=self.args.train_group_size)
        idx = random.randint(0, self.args.train_group_size - 1)
        negs[idx] = pos_pid
        group_batch = []
        for iid in negs:
            title = self.pid2title.get(iid)
            if title == '-':
                title = 'null'
            psg = title + ', text: ' + self.pid2txt[iid]
            item = self.create_one_example(psg)
            item['label'] = idx
            group_batch.append(item)
        return encoded_query, group_batch

class GroupedTrainQA(Dataset):
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

    def create_one_example(self, doc_encoding: str, max_len:int):
        item = self.tok.encode_plus(
            doc_encoding.strip(),
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        return item

    def cut_words(self, psg):
        if len(psg) >= self.args.p_max_len:
            return psg
        newid = random.randint(0, self.total_len - 10)
        group = self.nlp_dataset[newid]
        pos_pid = random.choice(group['pos'])
        text = self.idx2txt[int(pos_pid)]
        remain = self.args.p_max_len - len(psg)
        idx = random.randint(1, 4)
        if idx == 1:
            return psg
        elif idx == 2:
            start = random.randint(0,3)
            return psg[start:]
        elif idx == 3:
            start = random.randint(0,20)
            psg = text[start:remain+start] + psg
            return psg
        elif idx == 4:
            if random.randint(0, 1) == 0:
                psg = text[:remain] + psg
            else:
                psg = psg + text[:remain]
            return psg
        return psg

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qtext = group['qry']
        pos_pid = random.choice(group['pos'])
        # select  = self.nlp_dataset[random.randint(0, self.__len__())]
        neg_group = group['neg'] #+ select['neg'][:3]
        if len(neg_group) < self.args.train_group_size:
            negs = random.choices(neg_group, k=self.args.train_group_size)
        else:
            negs = random.sample(neg_group, k=self.args.train_group_size)
        negs[0] = pos_pid
        group_batch = []
        encoded_query = self.create_one_example(qtext, self.args.q_max_len)
        for neg_id in negs:
            psg = self.idx2txt[int(neg_id)]
            psg = self.cut_words(psg)
            group_batch.append(self.create_one_example(psg, self.args.p_max_len))
        return encoded_query, group_batch

class GroupedTrainLine(Dataset):
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
        self.args = args
        self.args.train_group_size = 2
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args


    def __len__(self):
        return self.total_len

    def create_one_example(self, doc_encoding: str, max_len:int):
        item = self.tok.encode_plus(
            doc_encoding.strip(),
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]['text'].split('\t')
        qtext = group[0]
        group_batch = []
        encoded_query = self.create_one_example(qtext, self.args.q_max_len)
        assert len(group[2]) > 0  and len(group[4]) > 0
        for ptext in [group[2], group[4]]:
            group_batch.append(self.create_one_example(ptext, self.args.p_max_len))
        return encoded_query, group_batch


class PredictionQA(Dataset):
    columns = [
        'qid', 'pid', 'qry', 'psg'
    ]

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, q_max_len=32, p_max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
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

    def create_one_example(self, doc_encoding: str, max_len: int):
        item = self.tok.encode_plus(
            doc_encoding.strip(),
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]['text'].split('\t')
        qtext = group[0]
        neg_id = group[1]
        return self.create_one_example(qtext, self.q_max_len), self.create_one_example(self.idx2txt[int(neg_id)], self.p_max_len)

class RetrivalPredictionDataset(Dataset):
    columns = [
        'qid', 'pid', 'qry', 'psg'
    ]

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, q_max_len=32, p_max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.args = args
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
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

    def create_one_example(self, text: str, is_query=False):
        item = self.tok.encode_plus(
            text,
            truncation='only_first',
            return_attention_mask=False,
            max_length=self.q_max_len if is_query else self.p_max_len,
        )
        return item

    def __getitem__(self, item):
        group = self.nlp_dataset[item]
        qid = group['qry']
        encoded_query = self.create_one_example(self.qid2txt[qid], is_query=True)
        neg_group = group['neg'][:self.args.train_group_size]  # [nid for nid in group['neg'] if nid not in adds]
        group_batch = []
        for iid in neg_group:
            title = self.pid2title.get(iid)
            if title == '-':
                title = 'null'
            psg = title + ', text: ' + self.pid2txt[iid]
            item = self.create_one_example(psg)
            group_batch.append(item)
        return encoded_query, group_batch

class RetrivalEncodeDataset(Dataset):
    columns = [
        'qid', 'pid', 'qry', 'psg'
    ]

    def __init__(self, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'text',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.max_len = max_len
        collection_file = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/collection.tsv'
        query_file = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/queries.all.tsv'
        # self.pid2txt = self.read_txt(collection_file)
        self.pid2txt = self.read_txt(query_file)

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            if len(items) != 2:
                print(line.strip())
            qid2txt[items[0]] = items[1]
        return qid2txt

    def __len__(self):
        return len(self.nlp_dataset)

    def create_one_example(self, text: str):
        item = self.tok.encode_plus(
            text,
            truncation='only_first',
            return_attention_mask=False,
            max_length=self.max_len
        )
        return item

    def __getitem__(self, item):
        line = self.nlp_dataset[item]['text']
        psg = self.pid2txt[line.strip()]
        return self.create_one_example(psg), self.create_one_example(psg)

class EventRetrivalTrainDataset(Dataset):
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

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def __len__(self):
        return self.total_len

    def create_one_example(self, text: str, is_query=False):
        item = self.tok.encode_plus(
            text,
            truncation='only_first',
            return_attention_mask=False,
            max_length=self.args.q_max_len if is_query else self.args.p_max_len,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qid = group['qry']
        encoded_query = self.create_one_example(qid, is_query=True)
        pos_pid = group['pos'][0]
        neg_group = group['neg'][:self.args.train_group_size-1]
        group_batch = []
        for iid in [pos_pid] + neg_group:
            item = self.create_one_example(iid)
            group_batch.append(item)
        return encoded_query, group_batch

class EventRetrivalPredictionDataset(Dataset):

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, q_max_len=32, p_max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.args = args
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def __len__(self):
        return len(self.nlp_dataset)

    def create_one_example(self, text: str, is_query=False):
        item = self.tok.encode_plus(
            text,
            truncation='only_first',
            return_attention_mask=False,
            max_length=self.q_max_len if is_query else self.p_max_len,
        )
        return item

    def __getitem__(self, item):
        group = self.nlp_dataset[item]
        qid = group['qry']
        encoded_query = self.create_one_example(qid, is_query=True)
        pos_pid = group['pos'][0]
        neg_group = group['neg'][:self.args.train_group_size-1]  # [nid for nid in group['neg'] if nid not in adds]
        group_batch = []
        for iid in [pos_pid] + neg_group:
            item = self.create_one_example(iid)
            group_batch.append(item)
        return encoded_query, group_batch
