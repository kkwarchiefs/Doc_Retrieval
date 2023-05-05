#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : check_du_retrival.py
# @Author: 罗锦文
# @Date  : 2023/5/4
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import pickle
import json
import random
import collections

def find_du_idx():
    root = '/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/'
    idx2zh = pickle.load(open(root + "passage_idx.pkl", 'rb'))
    zh2idx = {}
    for idx, ptext in enumerate(idx2zh):
        zh2idx[ptext] = idx
    qry2pos = collections.defaultdict(list)
    qry2neg = collections.defaultdict(list)
    for line in sys.stdin:
        items = line.strip().split('\t')
        qry = items[0]
        ptext = items[2]
        idx = zh2idx.get(ptext)
        if idx is None:
            continue
        if items[3] == '1':
            qry2pos[qry].append(idx)
        else:
            qry2neg[qry].append(idx)
    for k, v in qry2pos.items():
        print(k, v, qry2neg[k], sep='\t')


if __name__ == "__main__":
    qry2label = {}
    for line in open('see'):
        items = line.strip().split('\t')
        qry2label[items[0]] = items[1:]
    for line in sys.stdin:
        ins = json.loads(line)
        labels = qry2label.get(ins['qry'])
        if labels is not None:
            ins['pos'] = eval(labels[0])
            ins['neg'] = eval(labels[1])
            print(json.dumps(ins, ensure_ascii=False))
