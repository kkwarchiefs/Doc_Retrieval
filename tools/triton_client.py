# 需要安装以下环境，在线编码镜像 默认已经安装。
# pip3 install numpy
# pip3 install nvidia-pyindex
# pip3 install tritonclient[all]
import sys
import time

import numpy as np
import tritonclient.http as httpclient
from transformers import BertTokenizer, AutoTokenizer
model_name = "embedding_mul_onnx"  # 模型目录名/venus注册模型名称
address = "10.212.207.33:8000"  # 机器地址
triton_client = httpclient.InferenceServerClient(url=address)
rm_model_path = "/search/ai/jamsluo/passage_rank/du_task_output/ernie_base_g49_5e5/checkpoint-16000/"
rm_model_path = "/search/ai/jamsluo/passage_rank/du_task_output/ernie_base_g2_5e5"
rm_model_path = "/search/ai/pretrain_models/infoxlm-base/"
tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
import torch

def get_embedding_old(doc):
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

def get_embedding(doc):
    RM_input = tokenizer(doc, max_length=512, truncation=True, return_tensors="pt", padding=True)
    # print(RM_input)
    RM_batch = [torch.tensor(RM_input["input_ids"]).numpy(), torch.tensor(RM_input["attention_mask"]).numpy()]

    inputs = []
    inputs.append(httpclient.InferInput('input_ids', list(RM_batch[0].shape), 'INT64'))
    inputs.append(httpclient.InferInput('attention_mask', list(RM_batch[1].shape), 'INT64'))
    inputs[0].set_data_from_numpy(RM_batch[0])
    inputs[1].set_data_from_numpy(RM_batch[1])
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
    # step = docs.split('。')
    # parts = [step[i] for i in range(len(step))][:128]
    # docs = ' '.join(open(sys.argv[1]).readlines())
    # parts = break_sentence(docs)
    docs = ' '.join(open(sys.argv[1]).readlines())
    step = len(docs) // 500
    parts = [docs[i*500:(i+1)*500] for i in range(step)]
    # parts = ["导读：对于假日我们总是很期待，尤其是法定假日，法定节假日是指根据各国、各民族的风俗习惯或纪念要求，由国家法律统一规定的用以进行庆祝及度假的休息时间。那么，2016中国法定节假日一年共有多少天?2016全年法定节假日多少天?快点随万年历小编详细了解下中国法定节假日多少天吧。什么是法定假日法定节假日是指根据各国、各民族的风俗习惯或纪念要求，由国家法律统一规定的用以进行庆祝及度假的休息时间。法定节假日制度是国家政治、经济、文化制度的重要反映，涉及经济社会的多个方面，涉及广大人民群众的切身利益。2016中国法定节假日共有多少天",
    #          "一年国家法定节假日为11天。根据公布的国家法定节假日调整方案，调整的主要内容包括：元旦放假1天不变；春节放假3天，放假时间为农历正月初一、初二、初三；“五一”国际劳动节1天不变；“十一”国庆节放假3天；清明节、端午节、中秋节增设为国家法定节假日，各放假1天(农历节日如遇闰月，以第一个月为休假日)。3、允许周末上移下错，与法定节假日形成连休。",
    #          "我国制定的法定节假日都是在规定的时间之内,要求用人单位给员工带薪休假的。不过如果不休假的话,按时的支付加班费也可以。法定节假日肯定是在规定的时间之内放假的,我国每年的法定节假日总体算起来也是相当长的一段时间。那么按照我国的规定,国家法定假日一年多少天呢?一、国家法定假日一年多少天?一年中法定假日包括周六周日一共有115天或116天。计算方法:我国共有法定节假日11天(包括春节、国庆两个假期各3天,元旦、清明、五",
    #          "一年中国家法定节假日有哪些天中国的法定节假日是给上班族的休息时间,也是国家对上班族给予的休息权利。但是节假日的多少是有法律规定的,那么一年法定节假日多少天呢?法律对于劳动者休息的权利做了相应的规定,一周休息两天以及国家规定的一些重要节日的假期,一共有一百一十五天左右。详细内容华律网小编为你解答。"]
    parts = ["Several families of Byzantine Greece were of Norman mercenary origin during the period of the Comnenian Restoration, when Byzantine emperors were seeking out western European warriors. The Raoulii were descended from an Italo-Norman named Raoul, the Petraliphae were descended from a Pierre d'Aulps, and that group of Albanian clans known as the Maniakates were descended from Normans who served under George Maniaces in the Sicilian expedition of 1038.",
             "在圣埃夫鲁（Saint Evroul），歌唱的传统得到了发展，合唱团在诺曼底声名achieved起。在诺曼方丈罗伯特·德·格兰特梅尼尔（Robert de Grantmesnil）的统治下，圣埃夫鲁的几名僧侣逃往意大利南部，罗伯>特·吉斯卡德（Robert Guiscard）资助他们，并在圣欧菲米娅（Sant'Eufemia）建立了一座拉丁修道院。在那里，他们延续了唱歌的传统。",
             "拜占庭事态的进一步下降为1185年的第三次进攻铺平了道路，当时，由于背叛了拜占庭的高级官员，一支庞大的诺曼军队入侵了Dyrachium。一段时间后，Dy拉虫病（亚得里亚海最重要的海军基地之一）再次落入>拜占庭之手。"]
    embed = get_embedding(parts)
    while True:
        raw_text = input("\nContext prompt (stop to exit) >>> ")
        if not raw_text:
            print('Prompt should not be empty!')
            continue
        if raw_text == "stop":
            break
        # texts = raw_text.split('|')
        raw_embed = get_embedding(raw_text)
        scores = np.matmul(raw_embed, embed.transpose(1, 0))[0]
        sortid = np.argsort(-scores)
        for id in sortid[:5]:
            print(parts[id], scores[id])
