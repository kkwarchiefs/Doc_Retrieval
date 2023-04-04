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

if __name__ == "__main__":
    for line in sys.stdin:
        ins = json.loads(line)
        for a in [ins['pos'][0]] + ins['neg'][:3]:
            print(ins['qry'], a, sep='\t')
