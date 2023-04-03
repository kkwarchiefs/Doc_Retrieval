#!/bin/bash
export LC_ALL="en_US.utf8"
/data/miniconda3/condabin/conda activate env-3.6.8
/data/miniconda3/envs/env-3.6.8/bin/pip install transformers==4.2.1
/data/miniconda3/envs/env-3.6.8/bin/pip install datasets
cd /cfs/cfs-i125txtf/jamsluo/ft_local/apex-master
/data/miniconda3/envs/env-3.6.8/bin/pip install -v --disable-pip-version-check --no-cache-dir ./
output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/lichee_inter_rocketqa_cls
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0/
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/classify_rank_top100.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
code_dir=/cfs/cfs-i125txtf/jamsluo/lichee_retrieval/
logs=${code_dir}/logs/lichee_inter_rocketqa_cls
cd ${code_dir}
/data/miniconda3/envs/env-3.6.8/bin/python -m torch.distributed.launch --nproc_per_node 4  run_interact.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_train \
  --save_steps 4000 \
  --train_dir $train_data_dir \
  --max_len 128 \
  --seed 66 \
  --fp16 \
  --adafactor \
  --per_device_train_batch_size 4 \
  --train_group_size 100 \
  --per_device_eval_batch_size 8 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy steps \
  --eval_steps 4000 \
  --pred_path $pred_path \
  --pred_id_file $pred_id_file >$logs 2>$logs
