# 需要安装以下环境，在线编码镜像 默认已经安装。
# pip3 install numpy
# pip3 install nvidia-pyindex
# pip3 install tritonclient[all]
import sys
import time

import numpy
import numpy as np
import tritonclient.http as httpclient
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, models, util
word_embedding_model = models.Transformer('/search/ai/pretrain_models/sbert-base-chinese-nli', do_lower_case=True)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

def get_embedding(texts):
    emb_list = embedder.encode(texts)
    # print(numpy.array(emb_list))
    return numpy.array(emb_list)


if __name__ == '__main__':
    docs = ' '.join(open(sys.argv[1]).readlines())
    # step = docs.split('，')
    step = len(docs) // 64
    parts = [docs[i*64:(i+2)*64] for i in range(step)]
    # parts = ["导读：对于假日我们总是很期待，尤其是法定假日，法定节假日是指根据各国、各民族的风俗习惯或纪念要求，由国家法律统一规定的用以进行庆祝及度假的休息时间。那么，2016中国法定节假日一年共有多少天?2016全年法定节假日多少天?快点随万年历小编详细了解下中国法定节假日多少天吧。什么是法定假日法定节假日是指根据各国、各民族的风俗习惯或纪念要求，由国家法律统一规定的用以进行庆祝及度假的休息时间。法定节假日制度是国家政治、经济、文化制度的重要反映，涉及经济社会的多个方面，涉及广大人民群众的切身利益。2016中国法定节假日共有多少天",
    #          "一年国家法定节假日为11天。根据公布的国家法定节假日调整方案，调整的主要内容包括：元旦放假1天不变；春节放假3天，放假时间为农历正月初一、初二、初三；“五一”国际劳动节1天不变；“十一”国庆节放假3天；清明节、端午节、中秋节增设为国家法定节假日，各放假1天(农历节日如遇闰月，以第一个月为休假日)。3、允许周末上移下错，与法定节假日形成连休。",
    #          "我国制定的法定节假日都是在规定的时间之内,要求用人单位给员工带薪休假的。不过如果不休假的话,按时的支付加班费也可以。法定节假日肯定是在规定的时间之内放假的,我国每年的法定节假日总体算起来也是相当长的一段时间。那么按照我国的规定,国家法定假日一年多少天呢?一、国家法定假日一年多少天?一年中法定假日包括周六周日一共有115天或116天。计算方法:我国共有法定节假日11天(包括春节、国庆两个假期各3天,元旦、清明、五",
    #          "一年中国家法定节假日有哪些天中国的法定节假日是给上班族的休息时间,也是国家对上班族给予的休息权利。但是节假日的多少是有法律规定的,那么一年法定节假日多少天呢?法律对于劳动者休息的权利做了相应的规定,一周休息两天以及国家规定的一些重要节日的假期,一共有一百一十五天左右。详细内容华律网小编为你解答。"]
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
        scores = util.cos_sim(raw_embed, embed)[0]
        scores = scores.numpy()
        # scores = np.sum(raw_embed * embed, axis=-1)
        sortid = np.argsort(-scores)
        for id in sortid[:5]:
            print(parts[id], scores[id])
