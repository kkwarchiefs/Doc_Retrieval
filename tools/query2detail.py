#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : query2detail.py
# @Author: 罗锦文
# @Date  : 2023/4/7
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import json
import re
def read_rocket():
    id2pos = {}
    id2neg = {}
    for line in open('./tempdata/rocket_rand128.json'):
        ins = json.load(line)
        id2pos[ins['qry']] = ins['pos']
        id2neg[ins['qry']] = ins['neg']
    for line in open('./tempdata/rocket_aug128.json'):
        ins = json.load(line)
        id2pos[ins['qry']] = list(set(id2pos[ins['qry']] + ins['pos']))
        id2neg[ins['qry']] = list(set(id2neg[ins['qry']] + ins['neg']))
    return id2pos, id2neg

if __name__ == "__main__":
    for line in sys.stdin:
        ins = json.loads(line)
        prompt = ins['prompt'].replace('\n\n', '\n').split('\n')[1:]
        label = eval(ins['label'])
        response = ins['response'].split('\n')
        if len(response) != 20:
            print(line.strip(), len(response), file=sys.stderr)
            continue
        for a,b,c in zip(label, prompt, response):
            b = re.sub('^\\d+\\.', '', b).strip()
            c = re.sub('^\\d+\\.', '', c).strip()
            print({'qry':a, 'en': b, 'zh': c})
