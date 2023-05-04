#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : convert_model.py
# @Author: 罗锦文
# @Date  : 2023/3/14
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import torch
import os

def convert_model():
    RM_model_path = sys.argv[1]
    model_dict = torch.load(os.path.join(RM_model_path, 'pytorch_model.bin'), map_location="cpu")
    for k, _ in model_dict.items():
        print(k)
    new_model_dict = {k.replace('hf_', ''): v for k, v in model_dict.items()}
    torch.save(new_model_dict, os.path.join(RM_model_path, 'pytorch_model.bin'))

def read_model():
    RM_model_path = sys.argv[1]
    model_dict = torch.load(os.path.join(RM_model_path, 'pytorch_model.bin'), map_location="cpu")
    print([k for k, v in model_dict.items()])

if __name__ == "__main__":
    convert_model()
