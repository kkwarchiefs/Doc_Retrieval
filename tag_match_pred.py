import argparse
import os
from string import Template
import sys
import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoConfig, AutoModel,AutoTokenizer
from transformers import BertTokenizer
parser = argparse.ArgumentParser()

# parser.add_argument("--device", type=int, default=0)
# args = parser.parse_args()

model_name = "tag_match_onnx"
device = torch.device('cuda:0')

class RetrieverInfer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        model = cls(hf_model)
        return model

    def forward(self, input_ids, attention_mask):
        doc_input = {}
        doc_input['input_ids'] = input_ids.to(self.model.device)
        doc_input['attention_mask'] = attention_mask.to(self.model.device)
        doc_out = self.model(**doc_input, return_dict=True)
        doc_token_embeddings = doc_out.last_hidden_state
        doc_input_mask_expanded = doc_input['attention_mask'].unsqueeze(-1).expand(doc_token_embeddings.size()).float()
        doc_cls = torch.sum(doc_token_embeddings * doc_input_mask_expanded, 1) / torch.clamp(doc_input_mask_expanded.sum(1), min=1e-9)
        return doc_cls


RM_model_path = "/search/ai/jamsluo/passage_rank/du_task_output/ernie_base_g49_5e5/checkpoint-16000/"
RM_model_path = "/search/ai/pretrain_models/Dense-bert_base-contrast-dureader/"
RM_model_path = "/search/ai/jamsluo/passage_rank/du_task_output//ernie_base_g2_5e5_dureader_train/checkpoint-4000/"
RM_model_path = sys.argv[1]

RM_tokenizer = AutoTokenizer.from_pretrained(RM_model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(
		RM_model_path,
        trust_remote_code=True
    )

RM_model = RetrieverInfer.from_pretrained(RM_model_path, config=config, trust_remote_code=True)
RM_model = RM_model.to(device)
RM_model = RM_model.eval()  # 转换为eval模式


for line in open(sys.argv[2]):
    response_text = '服用三七粉期间,孕妇和儿童不宜使用。 三七粉是处方药,不是药品。 过量服用会引起中毒。'
    temp_inputs = RM_tokenizer([response_text, response_text[:10]], max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
    if 'token_type_ids' in temp_inputs:
        temp_inputs.pop('token_type_ids')
    inputs = (temp_inputs['input_ids'], temp_inputs['attention_mask'])  # 模型测试输入数据
    RM_model = RM_model.eval()  # 转换为eval模式
    res = RM_model(**temp_inputs)
    for vec in res:
        print(vec)

