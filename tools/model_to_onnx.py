import argparse
import os
from string import Template

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers import BertTokenizer
parser = argparse.ArgumentParser()

parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

model_name = "embedding_passage_onnx"
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

    def forward(self, doc_input):
        doc_input['input_ids'] = doc_input['input_ids'].to(self.model.device)
        if 'token_type_ids' in doc_input:
            doc_input['token_type_ids'] = doc_input['token_type_ids'].to(self.model.device)
        doc_input['attention_mask'] = doc_input['attention_mask'].to(self.model.device)
        doc_out = self.model(**doc_input, return_dict=True)
        return doc_out.last_hidden_state[:, 0]


RM_model_path = "/search/ai/jamsluo/passage_rank/du_task_output/roberta_large_g9_1e5/checkpoint-9000"

RM_tokenizer = BertTokenizer.from_pretrained(RM_model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(
		RM_model_path,
        trust_remote_code=True
    )

RM_model = RetrieverInfer.from_pretrained(RM_model_path, config=config, trust_remote_code=True)
# model_dict = torch.load(os.path.join(RM_model_path, 'pytorch_model.bin'), map_location="cpu")
# new_model_dict = {k.replace('hf_model.', ''): v for k, v in model_dict.items()}
# load_result = RM_model.load_state_dict(new_model_dict, strict=True)

RM_model = RM_model.half().to(device)
response_text = '服用三七粉期间,孕妇和儿童不宜使用。 三七粉是处方药,不是药品。 过量服用会引起中毒。'
temp_inputs = RM_tokenizer(response_text + response_text, max_length=512, truncation=True, return_tensors="pt").to(device)
print(temp_inputs)
inputs = (temp_inputs['input_ids'], temp_inputs['token_type_ids'], temp_inputs['attention_mask'])  # 模型测试输入数据

# print(RM_model(input['input_ids'], input['attention_mask']))
RM_model = RM_model.eval()  # 转换为eval模式
os.makedirs(f"/search/ai/jamsluo/passage_rank/du_task_output/model_store/{model_name}/1", exist_ok=True)
torch.onnx.export(
	RM_model,
	inputs,
	f"model_store/{model_name}/1/model.onnx",  # 输出模型文件名
	input_names=['input_ids', 'token_type_ids', 'attention_mask'],  # 输入节点名，每一个名称对应一个输入名
    output_names=['output'],  # 输出节点名，每一个名称对应一个输出名
	opset_version=14,
	dynamic_axes={'input_ids': {0: 'B', 1: 'C'}, 'token_type_ids': {0: 'B', 1: 'C'}, 'attention_mask': {0: 'B', 1: 'C'}}  # 声明动态维度，默认输入维度固定，可以声明为可变维度（用字符串占位符表示）
)


# traced_script_module.save(f"model_store/{model_name}/1/traced-model.pt")



