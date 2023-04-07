# Copyright 2021 Reranker Author. All rights reserved.
# Code structure inspired by HuggingFace run_glue.py in the transformers library.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import sys
from retriever.modeling import *
from retriever.trainer import *
from retriever.data import *
from retriever.arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments
from collections import defaultdict
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
import time
import numpy as np
import pickle
import tqdm


@dataclass
class QryDocCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 128
    max_d_len: int = 512

    def __call__(
            self, features
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_d_len,
            return_tensors="pt",
        )

        return {'qry_input': q_collated, 'doc_input': d_collated}
os.environ["WANDB_DISABLED"] = "true"


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(training_args.output_dir, exist_ok=True)
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logging.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logging.info("Training/evaluation parameters %s", training_args)
    logging.info("Model parameters %s", model_args)
    logging.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        output_hidden_states=True,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    _model_class = RetrieverMean  # COILSentence
    _trainer_class = RerankerRetrival
    _train_class = GroupedTrainMUL
    _eval_class = PredictionPure
    _test_class = PredictionPure
    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model = model.to(training_args.device)

    # Get datasets
    def compute_metrics(p):
        formate_output(p.predictions, training_args.output_dir + '/' + str(int(time.time())) + '.tsv')
        return p.predictions

    def formate_output(pred_scores, outfile):
        pred_qids = []
        pred_pids = []
        with open(data_args.pred_path[0], 'r', encoding='utf-8') as f:
            for l in f:
                q, p = l.strip().split('\t')[:2]
                pred_qids.append(q)
                pred_pids.append(p)
        if trainer.is_world_process_zero():
            assert len(pred_qids) == len(pred_scores)
            all_scores = defaultdict(dict)
            for qid, did, score in zip(pred_qids, pred_pids, pred_scores):
                if did in all_scores[qid]:
                    all_scores[qid][did] = max(score, all_scores[qid][did])
                else:
                    all_scores[qid][did] = score
            qq = list(all_scores.keys())
            with open(outfile, "w") as writer:
                for qid in qq:
                    score_list = sorted(list(all_scores[qid].items()), key=lambda x: x[1], reverse=True)
                    for rank, (did, score) in enumerate(score_list):
                        # res_doc = eval_dataset.idx2txt[int(did)].strip()
                        writer.write(f'{qid}\t{did}\t{rank + 1}\t{score}\n')

    if training_args.do_train and data_args.pred_path is not None:
        train_dataset = _train_class(
            data_args, data_args.train_path, tokenizer=tokenizer, train_args=training_args)
        eval_dataset = _eval_class(
            data_args, data_args.pred_path, tokenizer=tokenizer, q_max_len=data_args.q_max_len, p_max_len=data_args.p_max_len)

        # Initialize our Trainer
        trainer = _trainer_class(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=QryDocCollator(
                tokenizer,
                max_q_len=data_args.q_max_len,
                max_d_len=data_args.p_max_len),
        )
    elif training_args.do_train:
        train_dataset = _train_class(
            data_args, data_args.train_path, tokenizer=tokenizer, train_args=training_args)
        trainer = _trainer_class(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            data_collator=QryDocCollator(
                tokenizer,
                max_q_len=data_args.q_max_len,
                max_d_len=data_args.p_max_len),
        )
    else:
        pass

    if training_args.do_train and trainer.is_world_process_zero():
        print(config)
        print(model_args)
        print(data_args)
        print(training_args)
        qry, doc = train_dataset[random.choice(range(train_dataset.__len__()))]
        input_ids = qry['input_ids']
        print(' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
        input_ids = doc[0]['input_ids']
        print(' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
        print(qry, doc)
        print('eval datasets')
        qry, doc = eval_dataset[random.choice(range(eval_dataset.__len__()))]
        input_ids = qry['input_ids']
        print(' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
        input_ids = doc['input_ids']
        print(' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
        print(qry, doc)
    # Training
    if training_args.do_train:
        # add 对抗训练
        # trainer.add_adv()
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        train_dataset = None
        eval_dataset = _eval_class(
            data_args, data_args.pred_path, tokenizer=tokenizer, q_max_len=data_args.q_max_len, p_max_len=data_args.p_max_len)
        trainer = _trainer_class(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=QryDocCollator(
                tokenizer,
                max_q_len=data_args.q_max_len,
                max_d_len=data_args.p_max_len),
        )
        trainer.evaluate()

    if training_args.do_encode:
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

        test_dataset = _test_class(data_args.pred_path, tokenizer=tokenizer, max_len=data_args.max_len)
        pred_lines = []
        for _path in data_args.pred_path:
            for line in open(_path, 'r', encoding='utf-8'):
                qid = line.strip().split('\t')[0]
                pred_lines.append(qid)
        pred_scores = trainer.predict(test_dataset=test_dataset).predictions
        if trainer.is_world_process_zero():
            assert len(pred_lines) == len(pred_scores)
            with open(data_args.rank_score_path, "wb") as writer:
                pickle.dump([pred_lines, pred_scores], writer)


if __name__ == "__main__":
    main()
