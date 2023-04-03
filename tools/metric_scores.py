#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : metric_scores.py
# @Author: 罗锦文
# @Date  : 2022/1/13
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
from collections import defaultdict
import os

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

def cal_file(file):
    group = defaultdict(list)
    for idx, line in enumerate(open(file)):
        if idx == 0 and 'question' in line:
            continue
        items = line.strip().split('\t')
        # print(line)
        group[items[0]].append((items[1], eval(items[3]), eval(items[4])))
    mapsum, mrrsum = 0, 0
    mapc, mrrc = 0, 0
    for k, v in group.items():
        score = calc_one_map(v)
        if score != 0:
            mapc += 1
        mapsum += score
        score = calc_one_mrr(v)
        if score != 0:
            mrrc += 1
        mrrsum += score
    print('map: ', mapsum/mapc, 'mrr: ', mrrsum/mrrc)

if __name__ == "__main__":
    input_path = sys.argv[1]
    files = os.listdir(input_path)
    for file_ in files:
        if file_.endswith('tsv'):
            print(file_)
            cal_file(os.path.join(input_path, file_))
