#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : convert_train_data.py
# @Author: 罗锦文
# @Date  : 2022/1/13
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
import sys
import codecs
from collections import defaultdict

def convert_train():
    corpus = defaultdict(list)
    count = 0
    for idx, line in enumerate(sys.stdin):
        if idx == 0:
            continue
        items = line.strip().split('\t')
        if len(items) != 3:
            print(line.strip, file=sys.stderr)
            continue
        corpus[items[0]].append([items[1], items[2]])
    for k, v in corpus.items():
        item = {
            'qry': k,
            'pos': [a[0] for a in v if a[1] == '1'],
            'neg': [a[0] for a in v if a[1] == '0']
        }
        if len(item['pos']) == 0:
            count += 1
        print(json.dumps(item))
    print(count, file=sys.stderr)

def trams():
    corpus = defaultdict(list)
    count = 0
    for idx, line in enumerate(sys.stdin):
        if idx == 0:
            continue
        items = line.strip().split('\t')
        if len(items) != 3:
            print(line.strip, file=sys.stderr)
            continue
        corpus[items[0]].append([items[1], items[2]])
    for k, v in corpus.items():
        item = {
            'qry': k,
            'psgs': [a[0] for a in v],
            'labels': [a[1] for a in v]
        }
        print(json.dumps(item))

def read_truth(files):
    qid2pid = defaultdict(list)
    for line in open(files):
        obj = json.loads(line)
        qid2pid[obj['question']] = [a['paragraph_id'] for a in obj['answer_paragraphs']]
    return qid2pid

if __name__ == "__main__":
    root = '/cfs/cfs-i125txtf/jamsluo/du_retrival_exp/DuReader-Retrieval-Baseline/dureader-retrieval-baseline-dataset/'
    truth = root + '/dureader_retrieval-data/train.json'
    p2id_map = root + 'passage-collection/passage2id.map.json'
    is_train = True
    with open(p2id_map, "r") as fr:
        pcid2pid = json.load(fr)
    pid2idx = {}
    for k,v in pcid2pid.items():
        pid2idx[v] = k
    topnumber = 100
    qid2pid = read_truth(truth)
    corpus = defaultdict(list)
    count = 0
    for idx, line in enumerate(sys.stdin):
        items = line.strip().split('\t')
        corpus[items[0]].append(items[1])
    for k, vlist in corpus.items():
        good = qid2pid[k]
        good_idx = [pid2idx[a] for a in good]
        if is_train:
            vlist = [a for a in vlist if a not in good_idx]
        # assert len(vlist) == topnumber
        item = {
            'qry': k,
            'pos': good_idx,
            'neg': vlist
        }
        if len(item['pos']) == 0:
            count += 1
        print(json.dumps(item))
        # for idx, v_ in enumerate(vlist):
        #     print(k, v_, idx+1, sep='\t', file=sys.stderr)
    print(count, file=sys.stderr)
