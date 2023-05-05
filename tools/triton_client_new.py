# 需要安装以下环境，在线编码镜像 默认已经安装。
# pip3 install numpy
# pip3 install nvidia-pyindex
# pip3 install tritonclient[all]
import sys
import numpy as np
import tritonclient.http as httpclient
from transformers import BertTokenizer, AutoTokenizer
model_name = "rerank_mul_onnx"  # 模型目录名/venus注册模型名称
address = "10.212.207.33:8000"  # 机器地址
triton_client = httpclient.InferenceServerClient(url=address)
rm_model_path = "/search/ai/pretrain_models/infoxlm-base/"
tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
import torch

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


if __name__ == '__main__':
    # step = docs.split('。')
    # parts = [step[i] for i in range(len(step))][:128]
    # docs = ' '.join(open(sys.argv[1]).readlines())
    # parts = break_sentence(docs)
    # torch.nn.Embedding
    # docs = ' '.join(open(sys.argv[1]).readlines())
    # step = len(docs) // 500
    # parts = [docs[i*500:(i+1)*500] for i in range(step)]
    # parts = ["导读：对于假日我们总是很期待，尤其是法定假日，法定节假日是指根据各国、各民族的风俗习惯或纪念要求，由国家法律统一规定的用以进行庆祝及度假的休息时间。那么，2016中国法定节假日一年共有多少天?2016全年法定节假日多少天?快点随万年历小编详细了解下中国法定节假日多少天吧。什么是法定假日法定节假日是指根据各国、各民族的风俗习惯或纪念要求，由国家法律统一规定的用以进行庆祝及度假的休息时间。法定节假日制度是国家政治、经济、文化制度的重要反映，涉及经济社会的多个方面，涉及广大人民群众的切身利益。2016中国法定节假日共有多少天",
    #          "一年国家法定节假日为11天。根据公布的国家法定节假日调整方案，调整的主要内容包括：元旦放假1天不变；春节放假3天，放假时间为农历正月初一、初二、初三；“五一”国际劳动节1天不变；“十一”国庆节放假3天；清明节、端午节、中秋节增设为国家法定节假日，各放假1天(农历节日如遇闰月，以第一个月为休假日)。3、允许周末上移下错，与法定节假日形成连休。",
    #          "我国制定的法定节假日都是在规定的时间之内,要求用人单位给员工带薪休假的。不过如果不休假的话,按时的支付加班费也可以。法定节假日肯定是在规定的时间之内放假的,我国每年的法定节假日总体算起来也是相当长的一段时间。那么按照我国的规定,国家法定假日一年多少天呢?一、国家法定假日一年多少天?一年中法定假日包括周六周日一共有115天或116天。计算方法:我国共有法定节假日11天(包括春节、国庆两个假期各3天,元旦、清明、五",
    #          "一年中国家法定节假日有哪些天中国的法定节假日是给上班族的休息时间,也是国家对上班族给予的休息权利。但是节假日的多少是有法律规定的,那么一年法定节假日多少天呢?法律对于劳动者休息的权利做了相应的规定,一周休息两天以及国家规定的一些重要节日的假期,一共有一百一十五天左右。详细内容华律网小编为你解答。"]
    # parts = ["Several families of Byzantine Greece were of Norman mercenary origin during the period of the Comnenian Restoration, when Byzantine emperors were seeking out western European warriors. The Raoulii were descended from an Italo-Norman named Raoul, the Petraliphae were descended from a Pierre d'Aulps, and that group of Albanian clans known as the Maniakates were descended from Normans who served under George Maniaces in the Sicilian expedition of 1038.",
    #          "在圣埃夫鲁（Saint Evroul），歌唱的传统得到了发展，合唱团在诺曼底声名achieved起。在诺曼方丈罗伯特·德·格兰特梅尼尔（Robert de Grantmesnil）的统治下，圣埃夫鲁的几名僧侣逃往意大利南部，罗伯>特·吉斯卡德（Robert Guiscard）资助他们，并在圣欧菲米娅（Sant'Eufemia）建立了一座拉丁修道院。在那里，他们延续了唱歌的传统。",
    #          "拜占庭事态的进一步下降为1185年的第三次进攻铺平了道路，当时，由于背叛了拜占庭的高级官员，一支庞大的诺曼军队入侵了Dyrachium。一段时间后，Dy拉虫病（亚得里亚海最重要的海军基地之一）再次落入>拜占庭之手。"]
    response_text = '服用三七粉期间,孕妇和儿童不宜使用。 三七粉是处方药,不是药品。 过量服用会引起中毒。'
    response_text = 'Q:如何提高绩效考核的有效性A:管理有效性的衡量管理的有效性足由效率和效果来衡量的.所谓效率是指投入与产出的比值.例如,设备利用率、工时利用率、劳动生产率、资金周转率以及单位生产成本等,这些是对组织效率的具体衡量.效果指日标>达成度,涉及活动的结果.效果的具体衡量指标有销售收人、利润额、销售利润率、成本利润率、资金利润率等.利润是销售收入与销售成本之间的差额,是衡量效果的一项客观指标.有效率无效果与有效果无效率都是不可取的.管理的有效性既追求效率又追求效果点击进入中大网校自学考试准题库（责任编辑：gx）'
    response_text = "Q:如何提高绩效考核的有效性A:绩效管理的有效性标准及方法目前绩效管理已经成为人力资源系统乃至整个组织的非常重要的管理职能。因为,绩效管理是开发并提高员工个体能力不可缺少的基础,是开发和甄选员工的工具和员工配置的基础,是确定培训对象和培训内容的基础,因此,绩效管理也是实现组织目标的重要手段和途径。 一、有效的绩效管理系统的标准 在美国,学者们曾对全美范围内3500家公司进行了一项调查,结果显示,最常被提及的人力资源管理功能是员工的绩效评价,但有30%～50%的员工认为,企业正规的绩效评价体系是无效的。因此,建立一个有效的绩效评价体系非常重要,而有效的绩效管理体系应具备战略一致性、效度、信度、公正性和明确性。 1.战略一致性。战略一致性是指绩效管理必须与组织战略、目标和文化相一致。战略一致性所强调的是,绩效管理系统需要为员工提供一种引导,从而使得员工能够为组织的成功做出贡献。这就需要绩效管理系统具有充分的弹性或敏感性,以适应公司战略的变化。事实上,尽管公司的战略重心多次发生转移,但大多数公司的绩效考评系统往往在相当长的时间内保持不变。然而,当公司的战略变化后,员工的工作行为也需要发生变化。如果公司的绩效考评系统没有随着公司战略的变化而变化,那么公司的绩效考评系统就很难正确评价员工的绩效。"
    parts = [response_text, response_text[:10]]
    embed = get_embedding(parts)
    print(embed)
    # while True:
    #     raw_text = input("\nContext prompt (stop to exit) >>> ")
    #     if not raw_text:
    #         print('Prompt should not be empty!')
    #         continue
    #     if raw_text == "stop":
    #         break
    #     # texts = raw_text.split('|')
    #     raw_embed = get_embedding(raw_text)
    #     scores = np.matmul(raw_embed, embed.transpose(1, 0))[0]
    #     sortid = np.argsort(-scores)
    #     for id in sortid[:5]:
    #         print(parts[id], scores[id])
