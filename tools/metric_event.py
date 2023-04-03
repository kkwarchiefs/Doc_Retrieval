#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : metric_event.py
# @Author: 罗锦文
# @Date  : 2022/2/8
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
from sklearn.metrics import classification_report

def calc_one_map(data):
    relcnt = 0
    score = 0.0
    data = sorted(data, key=lambda d: d[1], reverse=True)
    #print
    for idx, item in enumerate(data):
        #print idx
        #print item[0][2]
        #print item[1]
        if float(item[2]) == 1:
            relcnt = relcnt + 1
            score = score + 1.0 * relcnt / (idx + 1)
    if relcnt == 0:
        return 0
    return score / relcnt


def calc_one_mrr(data):
    score = 0
    data = sorted(data, key=lambda d: d[1], reverse=True)
    for idx, item in enumerate(data):
        if float(item[2]) == 1:
            score = 1.0 / (idx + 1)
            break
    return score


if __name__ == "__main__":
    mrr_all, map_all = 0, 0
    hold = float(sys.argv[2])
    y_true, y_pred = [], []
    for line in open(sys.argv[1]):
        items = line.strip().split('\t')
        qry, docs, scores = items[0], eval(items[1])[:4], eval(items[3])[:4]
        data = []
        for doc, score in zip(docs, scores):
            data.append([doc, score, 0])
        data[0][2] = 1
        y_ = [0] * len(scores)
        y_[0] = 1
        y_true += y_
        yy_ = []
        for a in scores:
            if a > hold:
                yy_.append(1)
            else:
                yy_.append(0)
        y_pred += yy_
        # map_all += calc_one_map(data)
        mrr_all += calc_one_mrr(data)
    print(mrr_all/1000)
    print(classification_report(y_true, y_pred))
