#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : corr.py
# @Author: 罗锦文
# @Date  : 2022/1/27
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
from scipy.stats import stats
from collections import defaultdict

def read_score(file):
    qid2list = defaultdict(list)
    for line in open(file):
        items = line.strip().split('\t')
        if len(items) == 4:
            # qid2list[items[0]].append((items[1], int(items[2]), float(items[3])))
            qid2list[items[0]].append(items[1])
        else:
            # qid2list[items[0]].append((items[1], int(items[2])))
            qid2list[items[0]].append(items[1])
    return qid2list

if __name__ == "__main__":
    a, b = read_score(sys.argv[1]), read_score(sys.argv[2])
    r_all, p_all = 0, 0
    for left, right in zip(a.values(), b.values()):
        right_reorder = [left.index(i) for i in right[:10]]
        r, p_value = stats.pearsonr(list(range(10)), right_reorder)
        r_all += r
        p_all += p_value
    print(r_all/len(a), p_all/len(a))
