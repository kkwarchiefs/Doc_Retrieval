# Copyright 2021 Reranker Author. All rights reserved.
# Code structure inspired by HuggingFace run_glue.py in the transformers library.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import sys
from reranker.modeling import *
from reranker.trainer import RerankerTrainer
from reranker.data import *
from reranker.arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments
from collections import defaultdict
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
import time
import codecs
import numpy as np
import json

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(training_args.output_dir, exist_ok=True)
    # fh = logging.FileHandler(training_args.output_dir + '/' + 'training.log')
    # fh.setLevel(logging.DEBUG)
    # global logger
    # logger.addHandler(fh)
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )


    logging.info("Training/evaluation parameters %s", training_args)
    logging.info("Model parameters %s", model_args)
    logging.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        # output_hidden_states=True,
        num_labels=num_labels,
        gradient_checkpointing=training_args.gradient_checkpointing,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    # if config.model_type == "gpt2":
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    # _model_class = RerankerDC if training_args.distance_cache else Reranker
    _model_class = ReClassifyGlobalMLP #ReClassifyGlobal
    _trainer_class = RerankerTrainer
    _train_class = GroupedTrainDatasetURLTitleGlobal
    _eval_class = PredictionDatasetGroupGlobal
    _test_class = PredictionDatasetGroupGlobal
    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model = model.to(training_args.device)
    # print('gradient_checkpointing', model.hf_model.config.gradient_checkpointing)
    # Get datasets
    def compute_metrics(p):
        formate_output(p.predictions, training_args.output_dir + '/' + str(int(time.time())) + '.tsv')
        return p.predictions

    def formate_output(pred_scores, outfile):
        pred_qids = []
        pred_pids = []
        score_split = data_args.train_group_size
        with open(data_args.pred_path[0], 'r', encoding='utf-8') as f:
            for l in f:
                ins = json.loads(l)
                pred_qids.append(ins['qry'])
                pred_pids.append(ins['neg'])
        if trainer.is_world_process_zero():
            assert len(pred_qids) == len(pred_scores)
            with open(outfile, "w") as writer:
                for qid, did, score in zip(pred_qids, pred_pids, pred_scores):
                    left = score[:score_split]
                    right = score[score_split:]
                    idx = np.argmax(left)
                    print(qid, did, idx, left.tolist(), right.tolist(), sep='\t', file=writer)

    if training_args.do_train and data_args.pred_path is not None:
        train_dataset = _train_class(
            data_args, data_args.train_path, tokenizer=tokenizer, train_args=training_args)
        eval_dataset = _eval_class(
            data_args, data_args.pred_path, tokenizer=tokenizer, max_len=data_args.max_len)


        # Initialize our Trainer
        trainer = _trainer_class(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=GroupCollator(tokenizer),
        )
    elif training_args.do_train:
        train_dataset = _train_class(
            data_args, data_args.train_path, tokenizer=tokenizer, train_args=training_args)
        trainer = _trainer_class(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            data_collator=GroupCollator(tokenizer),
        )
    else:
        train_dataset = None
        trainer = _trainer_class(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            data_collator=GroupCollator(tokenizer),
        )

    if training_args.do_train and trainer.is_world_process_zero():
        print(config)
        print(model_args)
        print(data_args)
        print(training_args)
        # for case in train_dataset[random.choice(range(train_dataset.__len__()))][:2]:
        #     input_ids = case['input_ids']
        #     print(' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
        #     print(case)
        # for case in eval_dataset[random.choice(range(eval_dataset.__len__()))]:
        #     input_ids = case['input_ids']
        #     print(' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
        #     print(case)
    # Training
    if training_args.do_train:
        # add 对抗训练
        # trainer.add_adv()
        trainer.train(
            model_path=model_args.reload_path if os.path.isdir(model_args.reload_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        trainer.evaluate()

    if training_args.do_predict:
        logging.info("*** Prediction ***")

        if os.path.exists(data_args.rank_score_path):
            if os.path.isfile(data_args.rank_score_path):
                raise FileExistsError(f'score file {data_args.rank_score_path} already exists')
            else:
                raise ValueError(f'Should specify a file name')
        else:
            score_dir = os.path.split(data_args.rank_score_path)[0]
            if not os.path.exists(score_dir):
                logging.info(f'Creating score directory {score_dir}')
                os.makedirs(score_dir)

        test_dataset = _test_class(data_args, data_args.pred_path, tokenizer=tokenizer, max_len=data_args.max_len)
        pred_scores = trainer.predict(test_dataset=test_dataset).predictions
        formate_output(pred_scores, data_args.rank_score_path)


if __name__ == "__main__":
    main()
