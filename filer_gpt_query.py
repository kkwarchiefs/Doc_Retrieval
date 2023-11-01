#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : filer_gpt_query.py
# @Author: 罗锦文
# @Date  : 2023/10/31
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs

if __name__ == "__main__":
    sec_set = set()
    for line in sys.stdin:
        items = line.strip().split('\t')
        if items[1] not in sec_set:
            print(line.strip())
        sec_set.add(items[1])
