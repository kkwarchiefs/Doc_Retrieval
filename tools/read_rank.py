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
if __name__ == "__main__":
    qry2best = {}
    for line in open('/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_all.top10'):
        items = line.strip().split('\t')
        if items[2] == '1':
            qry2best[items[0]] = items[1]
    qry2res = defaultdict(list)
    for line in sys.stdin:
        items = line.strip().split('\t')
        qry2res[items[0]].append((items[1], items[3]))
    score = 0
    detail = defaultdict(int)
    for k, vlist in qry2res.items():
        gold = qry2best[k]
        for id, v in enumerate(vlist):
            if v[0] == gold:
                score += 1/(id+1)
                detail[id+1] += 1
                print(k, '\t'.join(v), id+1, True, sep='\t')
            else:
                print(k, '\t'.join(v), id + 1, False, sep='\t')
    print('score:', score/len(qry2res), file=sys.stderr)
    print("detail:", detail, file=sys.stderr)

