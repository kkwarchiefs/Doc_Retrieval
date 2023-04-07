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
    qry2best = {}
    for line in open('/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev.json'):
        ins = json.loads(line)
        for a in [ins['pos'][0]] + ins['neg'][:9]:
            print(ins['qry'], a, sep='\t')

if __name__ == "__main__":
    idx2text = {}
    for line in sys.stdin:
        items = line.strip().split('\t')
        idx2text[int(items[0])] = items[1]
    pickle.dump(idx2text, open('en_passage_id.pkl', 'wb'))
