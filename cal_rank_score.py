#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : cal_rank_score.py
# @Author: 罗锦文
# @Date  : 2021/11/1
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import os
from collections import defaultdict
def score1():
    input_path = sys.argv[1]
    files = os.listdir(input_path)
    for file_ in files:
        if file_.endswith('tsv'):
            print(file_)
            os.system("python3 ms_marco_eval.py %s" % os.path.join(input_path, file_))


def merge_rank(finename):
    fout = open('outputs/tmp2','w')
    for line in open(finename):
        items = line.strip().split('\t')
        qry, neg, idx, score = items[0], eval(items[1]), int(items[2]), eval(items[3])
        tups = []
        for a, b in zip(neg, score):
            tups.append((a, b))
        # if score[0] < 0.1:
        tups.sort(key=lambda a: a[1], reverse=True)
        sort_ = [_a[0] for _a in tups]
        # sort_ = sort_ #+ neg[len(sort_):]
        # newdict = {a: idx+1 for idx, a in enumerate(neg[:len(sort_)])}
        # for idx, a in enumerate(sort_):
        #     newdict[a] += (idx+1)
        # merge = sorted(newdict.items(), key=lambda a:a[1])
        for i, _neg in enumerate(sort_[:10]):
            print(qry, _neg, i + 1, sep='\t',file=fout)
    fout.close()

def score2():
    input_path = sys.argv[1]
    files = os.listdir(input_path)
    for file_ in files:
        if file_.endswith('tsv'):
            print(file_)
            merge_rank(os.path.join(input_path, file_))
            os.system("python3 ms_marco_eval.py outputs/tmp2")

if __name__ == "__main__":
    score1()
