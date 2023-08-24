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
from sentence_transformers import SentenceTransformer

# parser.add_argument("--device", type=int, default=0)
# args = parser.parse_args()

model_name = "m3e_embedding_onnx"
device = torch.device('cuda:0')

RM_model_path = sys.argv[1]

RM_model = SentenceTransformer(RM_model_path, device='cpu')
class RetrieverInfer(nn.Module):
    def __init__(self, RM_model_path):
        super().__init__()
        self.model = SentenceTransformer(RM_model_path, device='cpu')

    def forward(self, input_ids, token_type_ids, attention_mask):
        doc_input = {}
        doc_input['input_ids'] = input_ids
        doc_input['token_type_ids'] = token_type_ids
        doc_input['attention_mask'] = attention_mask
        doc_out = self.model(doc_input)
        return doc_out['sentence_embedding']

RM_model = RM_model
response_text = '服用三七粉期间,孕妇和儿童不宜使用。 三七粉是处方药,不是药品。 过量服用会引起中毒。'
temp_inputs = RM_model.tokenize([response_text, response_text[:10]])
print(temp_inputs)
inputs = (temp_inputs['input_ids'], temp_inputs['token_type_ids'], temp_inputs['attention_mask'])  # 模型测试输入数据

# print(RM_model(input['input_ids'], input['attention_mask']))
RM_model = RM_model.eval()  # 转换为eval模式
print(RM_model(temp_inputs))
print(RM_model.encode([response_text, response_text[:10]]))
infer_model = RetrieverInfer(RM_model_path)
print(infer_model(inputs[0], inputs[1], inputs[2]))
os.makedirs(f"/search/ai/jamsluo/passage_rank/du_task_output/model_store/{model_name}/1", exist_ok=True)
torch.onnx.export(
	infer_model,
	inputs,
	f"/search/ai/jamsluo/passage_rank/du_task_output/model_store/{model_name}/1/model.onnx",  # 输出模型文件名
	input_names=['input_ids', 'token_type_ids', 'attention_mask'],  # 输入节点名，每一个名称对应一个输入名
    output_names=['output'],  # 输出节点名，每一个名称对应一个输出名
	opset_version=14,
	dynamic_axes={'input_ids': {0: 'B', 1: 'C'}, 'token_type_ids': {0: 'B', 1: 'C'}, 'attention_mask': {0: 'B', 1: 'C'}}  # 声明动态维度，默认输入维度固定，可以声明为可变维度（用字符串占位符表示）
)


# traced_script_module.save(f"model_store/{model_name}/1/traced-model.pt")



