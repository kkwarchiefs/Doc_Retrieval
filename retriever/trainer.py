# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Tuple, Optional, Any, Union

from .dist.sampler import SyncedSampler

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.utils.data.distributed import DistributedSampler

from transformers.trainer import Trainer, nested_detach
from transformers.trainer_utils import PredictionOutput, EvalPrediction, EvalLoopOutput
import logging
from apex import amp
logger = logging.getLogger(__name__)

class RerankerRetrival(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=state_dict):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _get_train_sampler(self):
        if self.args.local_rank == -1:
            return RandomSampler(self.train_dataset)
        elif self.args.collaborative:
            logger.info(f'Collaborative Mode.')
            return SyncedSampler(self.train_dataset, seed=self.args.seed)
        else:
            return DistributedSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # inputs['qry_input']['global_step'] = self.state.global_step
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if (self.state.global_step+1) % 1000 == 0: #and self.is_world_process_zero():
            logs: Dict[str, float] = {}
            if isinstance(outputs, dict):
                for k, v in outputs:
                    logs[k] = v.item()
            else:
                for idx, v in enumerate(outputs):
                    logs[str(idx)] = v.item()
            self.log(logs)
        return (loss, outputs) if return_outputs else loss


    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.warmup_ratio > 0:
            self.args.warmup_steps = num_training_steps * self.args.warmup_ratio
        # if self.optimizer is None:
        #     no_decay = ["bias", "LayerNorm.weight"]
        #     param_optimizer = list(self.model.named_parameters())
        #     first_group, second_group, other_group = [], [], []
        #     for n, p in param_optimizer:
        #         if not any(nd in n for nd in no_decay):
        #             if 'poly_' in n:
        #                 first_group.append(p)
        #             else:
        #                 second_group.append(p)
        #         else:
        #             other_group.append(p)
        #     optimizer_grouped_parameters = [
        #         {'params': first_group, 'lr': 3 * self.args.learning_rate, 'weight_decay_rate': self.args.weight_decay},
        #         {'params': second_group, 'weight_decay_rate': self.args.weight_decay},
        #         {'params': other_group, 'weight_decay_rate': 0.0}
        #     ]
        #     if self.args.adafactor:
        #         optimizer_cls = Adafactor
        #         optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        #     else:
        #         optimizer_cls = AdamW
        #         optimizer_kwargs = {
        #             "betas": (self.args.adam_beta1, self.args.adam_beta2),
        #             "eps": self.args.adam_epsilon,
        #         }
        #     optimizer_kwargs["lr"] = self.args.learning_rate
        #     self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return super(RerankerRetrival, self).create_optimizer_and_scheduler(num_training_steps)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    # def compute_metrics(self, p):
    #     import time
    #     import pickle
    #     print(self.args.output_dir + '/' + str(int(time.time())) + '.tsv')
    #     # formate_output(p.predictions, training_args.output_dir + '/' + str(int(time.time())) + '.tsv')
    #     pickle.dump(p.predictions, open(self.args.output_dir + '/' + str(int(time.time())) + '.pkl', 'wb'))
    #     return p.predictions

    # def compute_loss(self, model: Reranker, inputs):
    #     return model(inputs)['loss']

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        prepared = self._prepare_inputs(inputs)
        qry, doc = prepared['qry_input'], prepared['doc_input']
        # qry['global_step'] = self.state.global_step
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    outputs = model( qry, doc)
            else:
                outputs = model( qry, doc)

            loss = None
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        labels = None

        return (loss, logits, labels)

    def prediction_loop(
            self,
            *args,
            **kwargs
    ) -> EvalLoopOutput:
        pred_outs = super().prediction_loop(*args, **kwargs)
        preds, label_ids, metrics = pred_outs.predictions, pred_outs.label_ids, pred_outs.metrics
        preds = preds.squeeze()
        if self.compute_metrics is not None:
            metrics_no_label = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics_no_label = {}
        metrics_no_label = {}
        # for key in list(metrics_no_label.keys()):
        #     if not key.startswith("eval_"):
        #         metrics_no_label[f"eval_{key}"] = metrics_no_label.pop(key)

        return EvalLoopOutput(predictions=preds, label_ids=label_ids, metrics={**metrics, **metrics_no_label}, num_samples=preds.shape[0])
