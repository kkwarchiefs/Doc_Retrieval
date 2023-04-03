#!/bin/bash
export LC_ALL="en_US.utf8"
/data/miniconda3/condabin/conda activate env-3.6.8
/data/miniconda3/envs/env-3.6.8/bin/pip install transformers==4.2.1
/data/miniconda3/envs/env-3.6.8/bin/pip install datasets
cd /cfs/cfs-i125txtf/jamsluo/ft_local/apex-master
/data/miniconda3/envs/env-3.6.8/bin/pip install -v --disable-pip-version-check --no-cache-dir ./
output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/tmp
#init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0/
init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_pair_mlp_rocketqa/checkpoint-128000
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/classify_rank_top100.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
rank_score_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_pair_mlp_rocketqa/res.tsv
code_dir=/cfs/cfs-i125txtf/jamsluo/lichee_retrieval/
logs=${code_dir}/logs/lichee_inter_rocketqa_cls
cd ${code_dir}
/data/miniconda3/envs/env-3.6.8/bin/python -m torch.distributed.launch --nproc_per_node 4  run_interact_pair_mlp.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_predict \
  --max_len 128 \
  --seed 66 \
  --fp16 \
  --train_group_size 100 \
  --per_device_eval_batch_size 8 \
  --overwrite_output_dir \
  --pred_path $pred_path \
  --pred_id_file $pred_id_file \
  --rank_score_path $rank_score_path >$logs 2>$logs
