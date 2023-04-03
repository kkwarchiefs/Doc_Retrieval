# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle
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

from .arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments
import logging
from torch.autograd import Variable
from torch.nn import functional as F2
import os
from typing import Dict, List, Tuple, Iterable
from torch import Tensor

# from einops.layers.torch import Rearrange
from transformers import TrainingArguments, Trainer, BertTokenizer, BertModel, BertPreTrainedModel

logger = logging.getLogger(__name__)
from transformers.models.bert.modeling_bert import BertAttention


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F2.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class RankLoss(nn.Module):
    def forward(self, input):
        num = input.shape[-1]
        x = input.unsqueeze(2).expand(-1, -1, num)
        y = input.unsqueeze(1).expand(-1, num, -1)
        pair = torch.sigmoid(y - x)
        rr = 1. / (torch.sum(pair, dim=-1) + 0.5)
        loss = - torch.mean(rr[:, 0])
        return loss


class SmoothRank(nn.Module):

    def __init__(self):
        super(SmoothRank, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, scores):
        x_0 = scores.unsqueeze(dim=-1)  # [Q x D] --> [Q x D x 1]
        x_1 = scores.unsqueeze(dim=-2)  # [Q x D] --> [Q x 1 x D]
        diff = x_1 - x_0  # [Q x D x 1], [Q x 1 x D] --> [Q x D x D]
        is_lower = self.sigmoid(diff)  # [Q x D x D] --> [Q x D x D]
        ranks = torch.sum(is_lower, dim=-1) + 0.5  # [Q x D x D] --> [Q x D]
        return ranks


class SmoothMRRLoss(nn.Module):

    def __init__(self):
        super(SmoothMRRLoss, self).__init__()
        self.soft_ranker = SmoothRank()
        self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)

    def forward(self, scores, agg=True):
        ranks = self.soft_ranker(scores)  # [Q x D] --> [Q x D]
        # labels = torch.where(labels > 0, self.one, self.zero)                       # [Q x D] --> [Q x D]
        labels = self.one
        rr = labels / ranks  # [Q x D], [Q x D] --> [Q x D]
        rr_max, _ = rr.max(dim=-1)  # [Q x D] --> [Q]
        loss = 1 - rr_max  # [Q] --> [Q]
        if agg:
            loss = loss.mean()  # [Q] --> [1]
        return loss


class Reranker(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self._keys_to_ignore_on_save = []
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

    def forward(self, batch):
        batch['input_ids'] = batch['input_ids'].to(self.hf_model.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.hf_model.device)
        if 'token_type_ids' in batch:
            batch['token_type_ids'] = batch['token_type_ids'].to(self.hf_model.device)
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        # Add a clip to confine scores in [0, 1].
        # logits = torch.clamp(logits, 0, 1)
        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            loss = self.cross_entropy(scores, self.target_label)

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        # self.hf_model.save_pretrained(output_dir)
        path = os.path.join(output_dir, "pytorch_model.bin")
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), path)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

class RerankerEvent(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

    def forward(self, batch):
        batch['input_ids'] = batch['input_ids'].to(self.hf_model.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.hf_model.device)
        if 'token_type_ids' in batch:
            batch['token_type_ids'] = batch['token_type_ids'].to(self.hf_model.device)
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits
        scores = logits.view(
            self.train_args.per_device_train_batch_size,
            self.data_args.train_group_size
        )
        # Add a clip to confine scores in [0, 1].
        # logits = torch.clamp(logits, 0, 1)
        if self.training:

            loss = self.cross_entropy(scores, self.target_label)

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return scores

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class RerankerQA(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

    def forward(self, batch):
        if 'labels' in batch:
            tgt = batch.pop('labels').to(self.hf_model.device)
        batch['input_ids'] = batch['input_ids'].to(self.hf_model.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.hf_model.device)
        if 'token_type_ids' in batch:
            batch['token_type_ids'] = batch['token_type_ids'].to(self.hf_model.device)
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        # Add a clip to confine scores in [0, 1].
        # logits = torch.clamp(logits, 0, 1)
        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            loss = self.cross_entropy(scores, self.target_label)
            # loss = self.cross_entropy(logits, tgt)
            # convese = torch.cat([-scores[:, :1], scores[:, 1:]], dim=-1)
            # zero = torch.zeros_like(convese)
            # loss2 = torch.sum(torch.where(convese > zero, convese, zero))
            # loss = loss1 + 0.2 * loss2
            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class ReClassifyGlobalQA(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.mlp = nn.Linear(config.hidden_size * 2, 1)
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        mask = None
        group_size = None
        if self.training:
            group_size = self.data_args.train_group_size
        else:
            group_size = self.data_args.eval_group_size
        if 'labels' in batch:
            tgt = batch.pop('labels')
        if 'mask' in batch:
            mask = batch.pop('mask')
            mask = mask.view(-1, group_size)
        sent_out: BaseModelOutputWithPooling = self.hf_model(**batch, return_dict=True)
        reps = sent_out.last_hidden_state[:, 0]
        reps = reps.contiguous().view(
            -1,
            group_size,
            self.config.hidden_size,
        )
        left = reps.unsqueeze(2).expand(-1, -1, group_size, -1)
        right = reps.unsqueeze(1).expand(-1, group_size, -1, -1)
        reps_pair = torch.cat([left, right], dim=-1)
        reps_score = self.mlp(reps_pair).squeeze(dim=-1)

        if mask is not None:
            logit_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            reps_score = reps_score * logit_mask
        logits_sum1 = torch.sum(reps_score, dim=1)
        # score1 = F2.softmax(logits_sum1)
        logits_sum2 = torch.sum(reps_score, dim=2)
        # score2 = F2.softmax(-logits_sum2)
        score = logits_sum1
        if self.training:
            tgt = tgt.view(self.train_args.per_device_train_batch_size, group_size)
            tgt = tgt[:, 0]
            # vlook = logits_mtx.detach().numpy()
            # np.savez('vlook', score=vlook, tgt=tgt.numpy())
            # exit(-1)
            loss1 = self.cross_entropy(logits_sum1, tgt)
            loss2 = self.cross_entropy(-logits_sum2, tgt)
            loss = loss1 + loss2
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size

            return SequenceClassifierOutput(
                loss=loss,
                logits=score
            )
        else:
            return SequenceClassifierOutput(
                logits=score
            )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        path = args[0]
        reranker = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = reranker.load_state_dict(model_dict, strict=False)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('hf_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class RerankerPiont(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        ones_label = torch.zeros(self.train_args.per_device_train_batch_size, self.data_args.train_group_size,
                                 dtype=torch.long)
        ones_label[:, 0] = 1
        self.register_buffer(
            'ones_label',
            ones_label.view(-1)
        )
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        # print(batch['input_ids'].device, self.hf_model.device)
        # print(batch.keys())
        batch['input_ids'] = batch['input_ids'].to(self.hf_model.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.hf_model.device)
        if 'token_type_ids' in batch:
            batch['token_type_ids'] = batch['token_type_ids'].to(self.hf_model.device)
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits
        if self.training:
            loss = self.cross_entropy(logits, self.ones_label)
            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class RerankerMRR(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.focal_loss = FocalLoss(gamma=2, size_average=True)
        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )
        ones_label = torch.zeros(self.train_args.per_device_train_batch_size, self.data_args.train_group_size,
                                 dtype=torch.float)
        ones_label[:, 0] = 2
        ones_label[:, 1] = 1
        self.register_buffer(
            'ones_label',
            ones_label
        )
        self.mrr_loss = SmoothMRRLoss()
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits
        # Add a clip to confine scores in [0, 1].
        # logits = torch.clamp(logits, 0, 1)
        if self.model_args.temperature is not None:
            assert self.model_args.temperature > 0
            logits = logits / self.model_args.temperature

        if self.train_args.collaborative:
            logits = self.dist_gather_tensor(logits)
            logits = logits.view(
                self.world_size,
                self.train_args.per_device_train_batch_size,
                -1  # chunk
            )
            logits = logits.transpose(0, 1).contiguous()
        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            # loss = self.cross_entropy(scores, self.target_label)
            loss = self.mrr_loss(scores)
            # loss = self.mse_loss(scores, self.ones_label)
            # # print(loss, loss_add)
            # loss = loss_add + loss
            # loss = self.focal_loss(scores, self.target_label)
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, first, second, third, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(first, second)
        classifier_dropout = (
            hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(second, third)

    def forward(self, x, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ReClassifyLong(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = config
        self.classifier = nn.Linear(config.hidden_size, 1)
        self._keys_to_ignore_on_save = None
        self._keys_to_ignore_on_load_missing = None

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        # print(batch.keys())
        tgt = batch.pop('labels').to(self.hf_model.device)
        seg = batch.pop('seg').to(self.hf_model.device)
        batch['input_ids'] = batch['input_ids'].to(self.hf_model.device)
        # batch['token_type_ids'] = batch['token_type_ids'].to(self.hf_model.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.hf_model.device)
        sent_out: BaseModelOutputWithPooling = self.hf_model(**batch, return_dict=True)
        reps = sent_out.last_hidden_state
        # print(reps.size())
        seg_hidden = torch.gather(reps, 1, seg.unsqueeze(2).repeat(1, 1, self.config.hidden_size))
        # print(seg_hidden.size())
        logit = self.classifier(seg_hidden).squeeze(2)
        # print(logit.size())
        if self.training:
            loss = self.cross_entropy(logit, tgt)
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size

            return SequenceClassifierOutput(
                loss=loss,
                logits=logit
            )
        else:
            return SequenceClassifierOutput(
                logits=logit
            )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        path = args[0]
        reranker = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = reranker.load_state_dict(model_dict, strict=False)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('hf_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class ReClassifyLongMLP(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = config
        # self.classifier = nn.Linear(config.hidden_size, data_args.train_group_size)
        self.classifier = nn.Linear(config.hidden_size * data_args.train_group_size, data_args.train_group_size)
        # self.classifier = ClassificationHead(config.hidden_size * data_args.train_group_size, config.hidden_size,  data_args.train_group_size, config.hidden_dropout_prob)
        self._keys_to_ignore_on_save = None
        self._keys_to_ignore_on_load_missing = None

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        # print(batch.keys())
        tgt = batch.pop('labels').to(self.hf_model.device)
        seg = batch.pop('seg').to(self.hf_model.device)
        batch['input_ids'] = batch['input_ids'].to(self.hf_model.device)
        # batch['token_type_ids'] = batch['token_type_ids'].to(self.hf_model.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.hf_model.device)
        sent_out: BaseModelOutputWithPooling = self.hf_model(**batch, return_dict=True)
        reps = sent_out.last_hidden_state
        # logit = self.classifier(reps)
        # print(reps.size())
        seg_hidden = torch.gather(reps, 1, seg.unsqueeze(2).repeat(1, 1, self.config.hidden_size))
        # seg_hidden = seg_hidden.sum(-2)
        # print(seg_hidden.size())
        seg_hidden = seg_hidden.contiguous().view(-1, self.config.hidden_size * self.data_args.train_group_size)
        logit = self.classifier(seg_hidden)
        # print(logit.size())
        if self.training:
            loss = self.cross_entropy(logit, tgt)
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size

            return SequenceClassifierOutput(
                loss=loss,
                logits=logit
            )
        else:
            return SequenceClassifierOutput(
                logits=logit
            )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        path = args[0]
        reranker = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = reranker.load_state_dict(model_dict, strict=False)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('hf_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class ReClassifyPos(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = config
        self.classifier = nn.Linear(config.hidden_size, data_args.train_group_size)
        self._keys_to_ignore_on_save = None
        self._keys_to_ignore_on_load_missing = None

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        # print(batch.keys())
        tgt = batch.pop('labels').to(self.hf_model.device)
        seg = batch.pop('seg').to(self.hf_model.device)
        batch['input_ids'] = batch['input_ids'].to(self.hf_model.device)
        # batch['token_type_ids'] = batch['token_type_ids'].to(self.hf_model.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.hf_model.device)
        sent_out: BaseModelOutputWithPooling = self.hf_model(**batch, return_dict=True)
        reps = sent_out.last_hidden_state
        # print(reps.size())
        # seg_hidden = torch.gather(reps, 1, seg.unsqueeze(2).repeat(1, 1, self.config.hidden_size))
        # print(seg_hidden.size())
        logit = self.classifier(reps[:, 0, :])
        # print(logit.size())
        if self.training:
            loss = self.cross_entropy(logit, tgt)
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size

            return SequenceClassifierOutput(
                loss=loss,
                logits=logit
            )
        else:
            return SequenceClassifierOutput(
                logits=logit
            )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        path = args[0]
        reranker = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = reranker.load_state_dict(model_dict, strict=False)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('hf_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class ReClassifyMLP(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = config
        self.mlp = nn.Linear(self.data_args.train_group_size * config.hidden_size, self.data_args.train_group_size)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        if 'labels' in batch:
            tgt = batch.pop('labels')
        sent_out: BaseModelOutputWithPooling = self.hf_model(**batch, return_dict=True)
        # reps = torch.mean(sent_out.last_hidden_state, dim=1)
        reps = sent_out.last_hidden_state[:, 0]
        reps = reps.contiguous().view(
            -1,
            self.data_args.train_group_size * self.config.hidden_size
        )
        logit = self.mlp(reps)
        score = F2.softmax(logit)
        if self.training:
            tgt = tgt.view(self.train_args.per_device_train_batch_size, self.data_args.train_group_size)
            tgt = tgt[:, 0]
            loss = self.cross_entropy(logit, tgt)
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size

            return SequenceClassifierOutput(
                loss=loss,
                logits=score
            )
        else:
            return SequenceClassifierOutput(
                logits=score
            )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        path = args[0]
        reranker = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = reranker.load_state_dict(model_dict, strict=False)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('hf_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class ReClassifyGlobal(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.start = nn.Linear(config.hidden_size, config.hidden_size)
        self.end = nn.Linear(config.hidden_size, config.hidden_size)
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        if self.training:
            tgt = batch.pop('labels')
        sent_out: BaseModelOutputWithPooling = self.hf_model(**batch, return_dict=True)
        # reps = torch.mean(sent_out.last_hidden_state, dim=1)
        reps = sent_out.last_hidden_state[:, 0]
        rep_start = self.start(reps)
        rep_end = self.end(reps)
        rep_start = rep_start.contiguous().view(
            -1,
            self.data_args.train_group_size,
            self.config.hidden_size,
        )
        rep_end = rep_end.contiguous().view(
            -1,
            self.data_args.train_group_size,
            self.config.hidden_size,
        )
        logits = torch.einsum('bmd,bnd->bmn', rep_start, rep_end)
        logits_mtx = logits - torch.diag_embed(logits.diagonal(dim1=1, dim2=2))
        logits_sum1 = torch.sum(logits_mtx, dim=1)
        score1 = F2.softmax(logits_sum1)
        logits_sum2 = torch.sum(logits_mtx, dim=2)
        # score2 = F2.softmax(-logits_sum2)
        score = score1  # + score2
        if self.training:
            tgt = tgt.view(self.train_args.per_device_train_batch_size, self.data_args.train_group_size)
            tgt = tgt[:, 0]
            # vlook = logits_mtx.detach().numpy()
            # np.savez('vlook', score=vlook, tgt=tgt.numpy())
            # exit(-1)
            loss1 = self.cross_entropy(logits_sum1, tgt)
            loss2 = self.cross_entropy(-logits_sum2, tgt)
            loss = loss1 + loss2
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size

            return SequenceClassifierOutput(
                loss=loss,
                logits=score
            )
        else:
            return SequenceClassifierOutput(
                logits=score
            )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        path = args[0]
        reranker = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = reranker.load_state_dict(model_dict, strict=False)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('hf_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class ReClassifyGlobalHinge(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = config
        self.loss = nn.MultiMarginLoss()
        self.start = nn.Linear(config.hidden_size, config.hidden_size)
        self.end = nn.Linear(config.hidden_size, config.hidden_size)
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        if self.training:
            tgt = batch.pop('labels')
        sent_out: BaseModelOutputWithPooling = self.hf_model(**batch, return_dict=True)
        # reps = torch.mean(sent_out.last_hidden_state, dim=1)
        reps = sent_out.last_hidden_state[:, 0]
        rep_start = self.start(reps)
        rep_end = self.end(reps)
        rep_start = rep_start.contiguous().view(
            -1,
            self.data_args.train_group_size,
            self.config.hidden_size,
        )
        rep_end = rep_end.contiguous().view(
            -1,
            self.data_args.train_group_size,
            self.config.hidden_size,
        )
        logits = torch.einsum('bmd,bnd->bmn', rep_start, rep_end)
        logits_mtx = logits - torch.diag_embed(logits.diagonal(dim1=1, dim2=2))
        logits_sum1 = torch.sum(logits_mtx, dim=1)
        logits_sum2 = torch.sum(logits_mtx, dim=2)
        # score2 = F2.softmax(-logits_sum2)
        score = logits_sum1  # + score2
        if self.training:
            tgt = tgt.view(self.train_args.per_device_train_batch_size, self.data_args.train_group_size)
            tgt = tgt[:, 0]
            # vlook = logits_mtx.detach().numpy()
            # np.savez('vlook', score=vlook, tgt=tgt.numpy())
            # exit(-1)
            loss1 = self.loss(logits_sum1, tgt)
            loss2 = self.loss(-logits_sum2, tgt)
            loss = loss1 + loss2
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size

            return SequenceClassifierOutput(
                loss=loss,
                logits=score
            )
        else:
            return SequenceClassifierOutput(
                logits=score
            )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        path = args[0]
        reranker = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = reranker.load_state_dict(model_dict, strict=False)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('hf_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class ReClassifyAtt(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.trams_layer = BertAttention(config)
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.linear = nn.Linear(config.hidden_size, 1)
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        if 'labels' in batch:
            tgt = batch.pop('labels')
        sent_out: BaseModelOutputWithPooling = self.hf_model(**batch, return_dict=True)
        # reps = torch.mean(sent_out.last_hidden_state, dim=1)
        reps = sent_out.last_hidden_state[:, 0].contiguous().view(
            -1,
            self.data_args.train_group_size,
            self.config.hidden_size,
        )
        # cls_idx = torch.ones(reps.shape[0], dtype=torch.long, device=reps.device) * 101
        # pos_idx = torch.arange(self.data_args.train_group_size + 1, dtype=torch.long, device=reps.device) + 1
        # word_embeddings = self.hf_model.embeddings.word_embeddings
        # position_embedding = self.hf_model.embeddings.position_embeddings
        # cls_embeding = word_embeddings(cls_idx)
        # pos_embeding = position_embedding(pos_idx)
        # pos_embeding = pos_embeding.unsqueeze(0).expand(reps.shape[0], -1, -1)
        # reps = torch.cat([cls_embeding.unsqueeze(1), reps], dim=1)
        # reps = pos_embeding + reps
        att_res = self.trams_layer(reps)
        # norm_res = self.dropout(nn.GELU()(self.LayerNorm(att_res[0])))
        # att_reps = norm_res.contiguous().view(-1, self.data_args.train_group_size * self.config.hidden_size)
        logit = self.linear(att_res[0]).squeeze(2)
        # max_reps = torch.max(reps, dim=1).values
        # logit = self.poly_fc(max_reps)
        score = F2.softmax(logit)
        if self.training:
            # tgt = tgt['labels']
            tgt = tgt.view(self.train_args.per_device_train_batch_size, self.data_args.train_group_size)
            tgt = tgt[:, 0]
            loss = self.cross_entropy(logit, tgt)
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size

            return SequenceClassifierOutput(
                loss=loss,
                logits=score
            )
        else:
            return SequenceClassifierOutput(
                logits=score
            )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        path = args[0]
        reranker = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = reranker.load_state_dict(model_dict, strict=False)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('hf_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(4, 8, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 9 * 9, 240)
        self.fc2 = nn.Linear(240, 100)

    def forward(self, x):
        x = F2.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 14, 14)
        x = F2.relu(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        x = F2.relu(self.conv3(x))  # output(32, 10, 10)
        x = self.pool3(x)  # output(32, 5, 5)
        x = x.view(-1, 8 * 9 * 9)  # output(32*5*5)
        x = F2.relu(self.fc1(x))  # output(120)
        x = self.fc2(x)  # output(10)
        return x


class ReClassifyGlobalCNN(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = config
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.start = nn.Linear(config.hidden_size, config.hidden_size)
        self.end = nn.Linear(config.hidden_size, config.hidden_size)
        self.cnn = LeNet()
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        if self.training:
            tgt = batch.pop('labels')
        sent_out: BaseModelOutputWithPooling = self.hf_model(**batch, return_dict=True)
        # reps = torch.mean(sent_out.last_hidden_state, dim=1)
        reps = sent_out.last_hidden_state[:, 0]
        rep_start = self.start(reps)
        rep_end = self.end(reps)
        rep_start = rep_start.contiguous().view(
            -1,
            self.data_args.train_group_size,
            self.config.hidden_size,
        )
        rep_end = rep_end.contiguous().view(
            -1,
            self.data_args.train_group_size,
            self.config.hidden_size,
        )
        logits = torch.einsum('bmd,bnd->bmn', rep_start, rep_end)

        score = self.cnn(logits.unsqueeze(1))  # + score2
        if self.training:
            tgt = tgt.view(self.train_args.per_device_train_batch_size, self.data_args.train_group_size)
            tgt = tgt[:, 0]
            # vlook = logits_mtx.detach().numpy()
            # np.savez('vlook', score=vlook, tgt=tgt.numpy())
            # exit(-1)
            loss = self.loss(score, tgt)
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size
            return SequenceClassifierOutput(
                loss=loss,
                logits=score
            )
        else:
            return SequenceClassifierOutput(
                logits=score
            )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        path = args[0]
        reranker = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = reranker.load_state_dict(model_dict, strict=False)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('hf_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class ReClassifyGlobalMLP(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.mlp = nn.Linear(config.hidden_size * 2, 1)
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        if self.training:
            tgt = batch.pop('labels')
        sent_out: BaseModelOutputWithPooling = self.hf_model(**batch, return_dict=True)
        # reps = torch.mean(sent_out.last_hidden_state, dim=1)
        reps = sent_out.last_hidden_state[:, 0]
        reps = reps.contiguous().view(
            -1,
            self.data_args.train_group_size,
            self.config.hidden_size,
        )
        left = reps.unsqueeze(2).expand(-1, -1, self.data_args.train_group_size, -1)
        right = reps.unsqueeze(1).expand(-1, self.data_args.train_group_size, -1, -1)
        reps_pair = torch.cat([left, right], dim=-1)
        reps_score = self.mlp(reps_pair).squeeze(dim=-1)
        reps_score = reps_score - torch.diag_embed(reps_score.diagonal(dim1=1, dim2=2))
        vlook = reps_score.cpu().detach().numpy()
        np.savez('vlook_psg_dev', score=vlook)
        exit(-1)
        logits_sum1 = torch.sum(reps_score, dim=1)
        score1 = F2.softmax(logits_sum1)
        logits_sum2 = torch.sum(reps_score, dim=2)
        score2 = F2.softmax(-logits_sum2)
        score = torch.cat([score1, score2], dim=-1)
        if self.training:
            tgt = tgt.view(self.train_args.per_device_train_batch_size, self.data_args.train_group_size)
            tgt = tgt[:, 0]

            loss1 = self.cross_entropy(logits_sum1, tgt)
            loss2 = self.cross_entropy(-logits_sum2, tgt)
            loss = loss1 + loss2
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size
            return SequenceClassifierOutput(
                loss=loss,
                logits=score
            )
        else:
            return SequenceClassifierOutput(
                logits=score
            )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        path = args[0]
        reranker = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = reranker.load_state_dict(model_dict, strict=False)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('hf_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

class ReClassifyGlobalMLPActive(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, config):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        # self.mlp = nn.Linear(config.hidden_size * 2, 1)
        self.mlp = FeedForward(config.hidden_size * 2, config.hidden_size)
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        if self.training:
            tgt = batch.pop('labels')
        sent_out: BaseModelOutputWithPooling = self.hf_model(**batch, return_dict=True)
        # reps = torch.mean(sent_out.last_hidden_state, dim=1)
        reps = sent_out.last_hidden_state[:, 0]
        reps = reps.contiguous().view(
            -1,
            self.data_args.train_group_size,
            self.config.hidden_size,
        )
        left = reps.unsqueeze(2).expand(-1, -1, self.data_args.train_group_size, -1)
        right = reps.unsqueeze(1).expand(-1, self.data_args.train_group_size, -1, -1)
        reps_pair = torch.cat([left, right], dim=-1)
        reps_score = self.mlp(reps_pair).squeeze(dim=-1)
        reps_score = reps_score - torch.diag_embed(reps_score.diagonal(dim1=1, dim2=2))
        logits_sum1 = torch.sum(reps_score, dim=1)
        score1 = F2.softmax(logits_sum1)
        logits_sum2 = torch.sum(reps_score, dim=2)
        score2 = F2.softmax(-logits_sum2)
        score = torch.cat([score1, score2], dim=-1)
        if self.training:
            tgt = tgt.view(self.train_args.per_device_train_batch_size, self.data_args.train_group_size)
            tgt = tgt[:, 0]
            # vlook = reps_score.cpu().detach().numpy()
            # np.savez('vlook_psg', score=vlook, tgt=tgt.cpu().numpy())
            # exit(-1)
            loss1 = self.cross_entropy(logits_sum1, tgt)
            loss2 = self.cross_entropy(-logits_sum2, tgt)
            loss = loss1 + loss2
            # if self.train_args.collaborative or self.train_args.distance_cahce:
            # account for avg in all reduce
            # loss = loss.float() * self.world_size
            return SequenceClassifierOutput(
                loss=loss,
                logits=score
            )
        else:
            return SequenceClassifierOutput(
                logits=score
            )

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        path = args[0]
        reranker = cls(hf_model, model_args, data_args, train_args, kwargs['config'])
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = reranker.load_state_dict(model_dict, strict=False)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('hf_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
