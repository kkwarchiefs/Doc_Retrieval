# 需要安装以下环境，在线编码镜像 默认已经安装。
# pip3 install numpy
# pip3 install nvidia-pyindex
# pip3 install tritonclient[all]
import random
import sys
import time

import numpy as np
import tritonclient.http as httpclient
from transformers import BertTokenizer, AutoTokenizer
model_name = "embedding_passage_onnx"  # 模型目录名/venus注册模型名称
address = "10.212.207.33:8000"  # 机器地址
triton_client = httpclient.InferenceServerClient(url=address)
rm_model_path = "/search/ai/jamsluo/passage_rank/du_task_output/ernie_base_g49_5e5/checkpoint-16000/"
rm_model_path = "/search/ai/jamsluo/passage_rank/du_task_output/ernie_base_g2_5e5"
tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
import torch
import pickle

def get_embedding(doc):
    RM_input = tokenizer(doc, max_length=512, truncation=True, return_tensors="pt", padding=True)
    # print(RM_input)
    RM_batch = [torch.tensor(RM_input["input_ids"]).numpy(), torch.tensor(RM_input["token_type_ids"]).numpy(),  torch.tensor(RM_input["attention_mask"]).numpy()]

    inputs = []
    inputs.append(httpclient.InferInput('input_ids', list(RM_batch[0].shape), 'INT64'))
    inputs.append(httpclient.InferInput('token_type_ids', list(RM_batch[1].shape), 'INT64'))
    inputs.append(httpclient.InferInput('attention_mask', list(RM_batch[2].shape), 'INT64'))
    inputs[0].set_data_from_numpy(RM_batch[0])
    inputs[1].set_data_from_numpy(RM_batch[1])
    inputs[2].set_data_from_numpy(RM_batch[2])
    output = httpclient.InferRequestedOutput('output')
    # try:
    results = triton_client.infer(
        model_name,
        inputs,
        model_version='1',
        outputs=[output],
        request_id='1'
    )
    results = results.as_numpy('output')
    return results
    # print(results, sep='\t')

def break_sentence(text: str):
    # 转换成小写
    text = text.lower()
    ret = list()

    # 长度较短，不进行分句
    if len(text) < HALF_SENT_LEN:
        ret.append(text)
        return ret

    # 按照标点分句
    last_pos = 0
    for i, c in enumerate(text):
        if c in break_punctations and i - last_pos > MIN_SENT_LEN:
            segment = text[last_pos:i]
            if len(segment) > MAX_SENT_LEN:
                # 需要再次分句
                subs = force_break_sentence(segment)
                ret.extend(subs)
            else:
                ret.append(segment)

            # 分隔符
            last_pos = i + 1

    # 结尾仍有残余部分需要分句
    if last_pos < len(text):
        segment = text[last_pos:]
        if len(segment) > MAX_SENT_LEN:
            subs = force_break_sentence(segment)
            ret.extend(subs)
        else:
            ret.append(segment)

    return ret

PUNCT = 'w'

break_punctations = {'！', '？', '!', '?', '。'}
force_break_punctations = {',', ';', '，', '；'}
MAX_SENT_LEN = 512
HALF_SENT_LEN = 512
MIN_SENT_LEN = 256
# 按照小标点或长度强制分句
def force_break_sentence(text: str):
    ret = list()

    # 已经满足要求
    if len(text) <= MAX_SENT_LEN:
        ret.append(text)
        return ret

    idx = MAX_SENT_LEN - 1
    while idx >= 0 and text[idx] not in force_break_punctations:
        idx -= 1

    break_idx = idx + 1 if idx >= 0 else MAX_SENT_LEN
    ret.append(text[:break_idx])

    # 剩余部分
    remains = text[break_idx:]
    ret.extend(force_break_sentence(remains))
    return ret

if __name__ == '__main__':
    idx2txt = pickle.load(open(sys.argv[1], 'rb'))
    while True:
        raw_text = input("\nContext prompt (stop to exit) >>> ")
        if not raw_text:
            print('Prompt should not be empty!')
            continue
        if raw_text == "stop":
            break
        # texts = raw_text.split('|')
        parts = random.sample(range(len(idx2txt) - 10), k=128)
        docs = [idx2txt[i] for i in parts]
        embed = get_embedding(docs)
        raw_embed = get_embedding(raw_text)
        scores = np.matmul(raw_embed, embed.transpose(1, 0))[0]
        sortid = np.argsort(-scores)
        for id in sortid[:3]:
            print(docs[id], scores[id])
