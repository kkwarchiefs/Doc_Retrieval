#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_json.py
# @Author: 罗锦文
# @Date  : 2023/4/4
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import json
from collections import defaultdict
import pickle

def make_dev():
    root = '/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/'
    idx2zh =pickle.load(open(root + "passage_idx.pkl", 'rb'))
    for line in open('/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev.json'):
        ins = json.loads(line)
        for idx, a in enumerate([ins['pos'][0]] + ins['neg'][:9]):
            if idx == 0:
                print(ins['qry'], idx2zh[int(a)].replace('\t', ' '), 1, sep='\t')
            else:
                print(ins['qry'], idx2zh[int(a)].replace('\t', ' '), 0, sep='\t')

def make_multi():
    root = '/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/'
    idx2zh =pickle.load(open(root + "en_passage_idx.pkl", 'rb'))
    for line in open(sys.argv[1]):
        ins = json.loads(line)
        for idx, a in enumerate([ins['pos'][0]] + ins['neg'][:9]):
            if idx == 0:
                print(ins['zh'], idx2zh[int(a)].replace('\t', ' '), 1, sep='\t')
                print(ins['en'], idx2zh[int(a)].replace('\t', ' '), 1, sep='\t')
            else:
                print(ins['zh'], idx2zh[int(a)].replace('\t', ' '), 0, sep='\t')
                print(ins['en'], idx2zh[int(a)].replace('\t', ' '), 0, sep='\t')
if __name__ == "__main__":
    make_multi()
    # idx2text = {}
    # for line in sys.stdin:
    #     items = line.strip().split('\t')
    #     idx2text[int(items[0])] = items[1]
    # pickle.dump(idx2text, open('en_passage_id.pkl', 'wb'))
