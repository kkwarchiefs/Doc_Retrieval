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
        ins = json.loads(line)
        id2pos[ins['qry']] = ins['pos']
        id2neg[ins['qry']] = ins['neg']
    for line in open('./tempdata/rocket_aug128.json'):
        ins = json.loads(line)
        id2pos[ins['qry']] = list(set(id2pos[ins['qry']] + ins['pos']))
        id2neg[ins['qry']] = list(set(id2neg[ins['qry']] + ins['neg']))
    return id2pos, id2neg

def en_zh():
    id2pos, id2neg = read_rocket()
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
            if a not in id2pos:
                continue
            pos = id2pos[a]
            neg = id2neg[a]
            print(json.dumps({'qry':a, 'en': b, 'zh': c, "pos": pos, 'neg':neg},ensure_ascii=False))

if __name__ == "__main__":
    zh2en = {}
    for line in open(sys.argv[1]):
        ins = json.loads(line)
        prompt = ins['prompt'].replace('\n\n', '\n').split('\n')[1:]
        label = eval(ins['label'])
        response = ins['response'].replace('\n\n', '\n').split('\n')
        if len(response) != 10:
            print(line.strip(), len(response), file=sys.stderr)
            continue
        for a,b,c in zip(label, prompt, response):
            b = re.sub('^\\d{1,2}\\.', '', b).strip()
            c = re.sub('^\\d{1,2}\\.', '', c).strip()
            zh2en[b] = c
    count = 0
    for line in sys.stdin:
        ins = json.loads(line)
        ins['zh'] = ins['qry']
        ins['en'] = zh2en.get(ins['qry'], 'NULL')
        if ins['en'] == 'NULL':
            count += 1
        print(json.dumps(ins, ensure_ascii=False))
    print(count, file=sys.stderr)
