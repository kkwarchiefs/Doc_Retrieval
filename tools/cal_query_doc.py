#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : cal_query_doc.py
# @Author: 罗锦文
# @Date  : 2022/1/23
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs

if __name__ == "__main__":
    query_set = set()
    psg_set = set()
    for line in sys.stdin:
        items = line.strip().split('\t')
        query_set.add(items[0])
        psg_set.add(items[1])
    print('qry len', len(query_set))
    print('psg len', len(psg_set))
