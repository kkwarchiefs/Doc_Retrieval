#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : cal_rank_score.py
# @Author: 罗锦文
# @Date  : 2021/11/1
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
from collections import defaultdict

root = '/cfs/cfs-i125txtf/jamsluo/du_retrival_exp/DuReader-Retrieval-Baseline/'
q2id_map = root + "dureader-retrieval-baseline-dataset/dev/q2qid.dev.json"
p2id_map = root + "dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json"
MODEL_OUTPUT = "output/dev.res.top50"

outputf = 'outputs/dual_res.json'

# map query to its origianl ID
with open(q2id_map, "r") as fr:
    q2qid = json.load(fr)

# map para line number to its original ID
with open(p2id_map, "r") as fr:
    pcid2pid = json.load(fr)


def convert_json(recall_result):
    qprank = defaultdict(list)
    with open(recall_result, 'r') as f:
        for line in f.readlines():
            q, pcid, rank, score = line.strip().split('\t')
            qprank[q2qid[q]].append(pcid2pid[pcid])

    with open(outputf, 'w', encoding='utf-8') as fp:
        json.dump(qprank, fp, ensure_ascii=False, indent='\t')


def score1():
    input_path = sys.argv[1]
    files = os.listdir(input_path)
    for file_ in files:
        if file_.endswith('tsv'):
            print(file_)
            convert_json(os.path.join(input_path, file_))
            os.system(
                "python3 evaluation_du.py /cfs/cfs-i125txtf/jamsluo/du_retrival_exp/DuReader-Retrieval-Baseline/dureader-retrieval-baseline-dataset/dev/dev.json outputs/dual_res.json")


def merge_rank(finename):
    fout = open('outputs/tmp2', 'w')
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
            print(qry, _neg, i + 1, sep='\t', file=fout)
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
