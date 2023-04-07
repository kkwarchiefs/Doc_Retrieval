# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle
import random
from typing import Optional

import numpy as np
import torch
import torch.functional as F
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    PreTrainedModel, PreTrainedTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, \
    BaseModelOutputWithPastAndCrossAttentions
from torch import nn
import torch.distributed as dist
from transformers.models.bert.modeling_bert import BertAttention

from .arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments
import logging
from torch.autograd import Variable
from torch.nn import functional as F2
import os
from typing import Dict, List, Tuple, Iterable
from torch import Tensor

# from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)


class JointRetriever(nn.Module):
    def __init__(self, model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.model: PreTrainedModel = model
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.data_args, self.model_args, self.train_args = data_args, model_args, train_args
        self.config = config
        self.mlp = nn.Linear(config.hidden_size * 2, 1)

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        model = JointRetriever(hf_model, model_args, data_args, train_args, kwargs['config'])
        path = args[0]
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
        return model

    def save_pretrained(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def forward(self, qry_input: Dict, doc_input: Dict):
        group_size = self.data_args.train_group_size
        if 'label' in doc_input:
            labels = doc_input.pop('label').to(self.model.device)
        qry_input['input_ids'] = qry_input['input_ids'].to(self.model.device)
        if 'token_type_ids' in qry_input:
            qry_input['token_type_ids'] = qry_input['token_type_ids'].to(self.model.device)
        qry_input['attention_mask'] = qry_input['attention_mask'].to(self.model.device)
        doc_input['input_ids'] = doc_input['input_ids'].to(self.model.device)
        if 'token_type_ids' in doc_input:
            doc_input['token_type_ids'] = doc_input['token_type_ids'].to(self.model.device)
        doc_input['attention_mask'] = doc_input['attention_mask'].to(self.model.device)
        qry_out: BaseModelOutputWithPooling = self.model(**qry_input, return_dict=True)
        doc_out: BaseModelOutputWithPooling = self.model(**doc_input, return_dict=True)

        qry_cls = qry_out.last_hidden_state[:, 0]
        doc_cls = doc_out.last_hidden_state[:, 0]
        doc_reps = doc_cls.contiguous().view(
            -1,
            group_size,
            self.config.hidden_size,
        )
        ir_reps = qry_cls.unsqueeze(1) * doc_reps
        left = ir_reps.unsqueeze(2).expand(-1, -1, group_size, -1)
        right = ir_reps.unsqueeze(1).expand(-1, group_size, -1, -1)
        reps_pair = torch.cat([left, right], dim=-1)
        reps_score = self.mlp(reps_pair).squeeze(dim=-1)
        reps_score = reps_score - torch.diag_embed(reps_score.diagonal(dim1=1, dim2=2))
        # print(qry_cls[0][:10])
        # print(doc_cls[2][:10])
        # print(doc_cls[3][:10])
        # print(reps_pair.shape)
        # print(reps_pair[0][2][3][:10])
        # print(reps_pair[0][2][3][768:768+10])
        # print(reps_pair[0][2][3][768*2:768*2 +10])
        # print(reps_score)
        logits_sum1 = torch.sum(reps_score, dim=1)
        logits_sum2 = torch.sum(reps_score, dim=2)
        # print(logits_sum1)
        # print(logits_sum2)
        # print(labels)
        # exit(-1)

        if not self.training:
            score_ir = torch.sum(qry_cls.unsqueeze(1) * doc_reps, dim=-1)
            score1 = F2.softmax(logits_sum1)
            score2 = F2.softmax(-logits_sum2)
            score = torch.cat([score_ir, score1, score2], dim=-1)
            return score
        else:
            labels = labels.view(self.train_args.per_device_train_batch_size, self.data_args.train_group_size)
            labels = labels[:, 0]
            rank_loss = self.cross_entropy(logits_sum1, labels) + self.cross_entropy(-logits_sum2, labels)
            scores = torch.matmul(qry_cls, doc_cls.transpose(1, 0))  # Q * D
            index = torch.arange(
                scores.size(0),
                device=doc_input['input_ids'].device,
                dtype=torch.long
            )
            # offset the labels
            index = index * self.data_args.train_group_size
            label_cons = labels + index
            loss = self.cross_entropy(scores, label_cons)

            return rank_loss, loss, rank_loss


class JointRetrieverAtt(nn.Module):
    def __init__(self, model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.model: PreTrainedModel = model
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.data_args, self.model_args, self.train_args = data_args, model_args, train_args
        self.config = config
        self.trams_layer = BertAttention(config)
        self.mlp = nn.Linear(config.hidden_size, 1)

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        model = JointRetriever(hf_model, model_args, data_args, train_args, kwargs['config'])
        path = args[0]
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
        return model

    def save_pretrained(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def forward(self, qry_input: Dict, doc_input: Dict):
        group_size = self.data_args.train_group_size
        if 'label' in doc_input:
            labels = doc_input.pop('label').to(self.model.device)
        qry_input['input_ids'] = qry_input['input_ids'].to(self.model.device)
        if 'token_type_ids' in qry_input:
            qry_input['token_type_ids'] = qry_input['token_type_ids'].to(self.model.device)
        qry_input['attention_mask'] = qry_input['attention_mask'].to(self.model.device)
        doc_input['input_ids'] = doc_input['input_ids'].to(self.model.device)
        if 'token_type_ids' in doc_input:
            doc_input['token_type_ids'] = doc_input['token_type_ids'].to(self.model.device)
        doc_input['attention_mask'] = doc_input['attention_mask'].to(self.model.device)
        qry_out: BaseModelOutputWithPooling = self.model(**qry_input, return_dict=True)
        doc_out: BaseModelOutputWithPooling = self.model(**doc_input, return_dict=True)

        qry_cls = qry_out.last_hidden_state[:, 0]
        doc_cls = doc_out.last_hidden_state[:, 0]
        doc_reps = doc_cls.contiguous().view(
            -1,
            group_size,
            self.config.hidden_size,
        )
        reps = torch.cat([qry_cls.unsqueeze(1), doc_reps], dim=1)
        att_res = self.trams_layer(reps)
        logit = self.linear(att_res[0]).squeeze(2)
        if not self.training:
            score_ir = torch.sum(qry_cls.unsqueeze(1) * doc_reps, dim=-1)
            score_rank = F2.softmax(logit[:, 1:])
            score = torch.cat([score_ir, score_rank, score_rank], dim=-1)
            return score
        else:
            labels = labels.view(self.train_args.per_device_train_batch_size, self.data_args.train_group_size)
            labels = labels[:, 0]
            rank_loss = self.cross_entropy(logit[:, 1:], labels)
            scores = torch.matmul(qry_cls, doc_cls.transpose(1, 0))  # Q * D
            index = torch.arange(
                scores.size(0),
                device=doc_input['input_ids'].device,
                dtype=torch.long
            )
            # offset the labels
            index = index * self.data_args.train_group_size
            label_cons = labels + index
            loss = self.cross_entropy(scores, label_cons)

            return loss + rank_loss, loss, rank_loss


class RetrieverQA(nn.Module):
    def __init__(self, model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.model: PreTrainedModel = model
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.data_args, self.model_args, self.train_args = data_args, model_args, train_args

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        return model

    def save_pretrained(self, output_dir: str):
        path = os.path.join(output_dir, "pytorch_model.bin")
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), path)
        self.model.config.save_pretrained(output_dir)

    def forward(self, qry_input: Dict, doc_input: Dict):
        group_size = self.data_args.train_group_size
        if 'label' in doc_input:
            labels = doc_input.pop('label').to(self.model.device)
        qry_input['input_ids'] = qry_input['input_ids'].to(self.model.device)
        if 'token_type_ids' in qry_input:
            qry_input['token_type_ids'] = qry_input['token_type_ids'].to(self.model.device)
        qry_input['attention_mask'] = qry_input['attention_mask'].to(self.model.device)
        doc_input['input_ids'] = doc_input['input_ids'].to(self.model.device)
        if 'token_type_ids' in doc_input:
            doc_input['token_type_ids'] = doc_input['token_type_ids'].to(self.model.device)
        doc_input['attention_mask'] = doc_input['attention_mask'].to(self.model.device)
        qry_out: BaseModelOutputWithPooling = self.model(**qry_input, return_dict=True)
        doc_out: BaseModelOutputWithPooling = self.model(**doc_input, return_dict=True)

        qry_cls = qry_out.last_hidden_state[:, 0]
        doc_cls = doc_out.last_hidden_state[:, 0]

        if not self.training:
            score_ir = torch.sum(qry_cls * doc_cls, dim=-1)
            return score_ir
        else:

            scores = torch.matmul(qry_cls, doc_cls.transpose(1, 0))  # Q * D
            index = torch.arange(
                scores.size(0),
                device=doc_input['input_ids'].device,
                dtype=torch.long
            )
            # offset the labels
            index = index * self.data_args.train_group_size
            loss = self.cross_entropy(scores, index)
            return loss, loss


class Retriever(nn.Module):
    def __init__(self, model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.model: PreTrainedModel = model
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.data_args, self.model_args, self.train_args = data_args, model_args, train_args
        self.config = config

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        path = args[0]
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
        return model

    def save_pretrained(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def forward(self, qry_input: Dict, doc_input: Dict):
        group_size = self.data_args.train_group_size
        if 'label' in doc_input:
            labels = doc_input.pop('label').to(self.model.device)
        qry_input['input_ids'] = qry_input['input_ids'].to(self.model.device)
        if 'token_type_ids' in qry_input:
            qry_input['token_type_ids'] = qry_input['token_type_ids'].to(self.model.device)
        qry_input['attention_mask'] = qry_input['attention_mask'].to(self.model.device)
        doc_input['input_ids'] = doc_input['input_ids'].to(self.model.device)
        if 'token_type_ids' in doc_input:
            doc_input['token_type_ids'] = doc_input['token_type_ids'].to(self.model.device)
        doc_input['attention_mask'] = doc_input['attention_mask'].to(self.model.device)
        qry_out: BaseModelOutputWithPooling = self.model(**qry_input, return_dict=True)
        doc_out: BaseModelOutputWithPooling = self.model(**doc_input, return_dict=True)

        qry_cls = qry_out.last_hidden_state[:, 0]
        doc_cls = doc_out.last_hidden_state[:, 0]
        doc_reps = doc_cls.contiguous().view(
            -1,
            group_size,
            self.config.hidden_size,
        )
        if not self.training:
            score_ir = torch.sum(qry_cls.unsqueeze(1) * doc_reps, dim=-1)
            return score_ir
        else:
            scores = torch.matmul(qry_cls, doc_cls.transpose(1, 0))  # Q * D
            index = torch.arange(
                scores.size(0),
                device=doc_input['input_ids'].device,
                dtype=torch.long
            )
            # offset the labels
            index = index * self.data_args.train_group_size
            loss = self.cross_entropy(scores, index)
            return loss, loss

class RetrieverCosine(nn.Module):
    def __init__(self, model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.model: PreTrainedModel = model
        # self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.data_args, self.model_args, self.train_args = data_args, model_args, train_args
        self.config = config
        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        path = args[0]
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
        return model

    def save_pretrained(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def forward(self, qry_input: Dict, doc_input: Dict):
        group_size = self.data_args.train_group_size
        if 'label' in doc_input:
            labels = doc_input.pop('label').to(self.model.device)
        qry_input['input_ids'] = qry_input['input_ids'].to(self.model.device)
        if 'token_type_ids' in qry_input:
            qry_input['token_type_ids'] = qry_input['token_type_ids'].to(self.model.device)
        qry_input['attention_mask'] = qry_input['attention_mask'].to(self.model.device)
        doc_input['input_ids'] = doc_input['input_ids'].to(self.model.device)
        if 'token_type_ids' in doc_input:
            doc_input['token_type_ids'] = doc_input['token_type_ids'].to(self.model.device)
        doc_input['attention_mask'] = doc_input['attention_mask'].to(self.model.device)
        qry_out: BaseModelOutputWithPooling = self.model(**qry_input, return_dict=True)
        doc_out: BaseModelOutputWithPooling = self.model(**doc_input, return_dict=True)

        qry_cls = qry_out.last_hidden_state[:, 0]
        doc_cls = doc_out.last_hidden_state[:, 0]
        # qry_cls = nn.functional.normalize(qry_cls, dim=-1)
        # doc_cls = nn.functional.normalize(doc_cls, dim=-1)
        qry_expand = qry_cls.unsqueeze(1).expand(-1, group_size, -1)
        doc_reps = doc_cls.contiguous().view(
            -1,
            group_size,
            self.config.hidden_size,
        )
        euclidean_distance = torch.cosine_similarity(qry_expand, doc_reps, dim=-1)

        if not self.training:
            # score_ir = torch.sum(qry_cls.unsqueeze(1) * doc_reps, dim=-1)
            return euclidean_distance
        else:
            # index = -torch.ones(
            #     euclidean_distance.size(),
            #     device=doc_input['input_ids'].device,
            #     dtype=torch.float
            # )
            # index[:, 0] = 1
            # print(index, index.shape)
            # print(euclidean_distance, euclidean_distance.shape)

            # index = index.reshape(-1, 1)
            # pred = euclidean_distance.reshape(-1, 1)
            # qry_left = qry_expand.reshape(-1, self.config.hidden_size)
            # doc_right = doc_reps.reshape(-1, self.config.hidden_size)
            # offset the labels
            # if random.randint(0, 100) == 0:
            #     print(euclidean_distance)
            loss = self.criterion(euclidean_distance*10, self.target_label)
            return loss, loss
