#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : temp.py.py
# @Author: 罗锦文
# @Date  : 2022/1/24
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import json
a = ["1050857","789292","530572","1054923","1056726","270642","794665","1094191","1060391","482666","1049955","1051223","1051372","1091234","1051886","1052948","528760","529090","4947","918424"]

if __name__ == "__main__":
    for line in sys.stdin:
        obj = json.loads(line)
        if obj['qry'] in a:
            print(line.strip())
