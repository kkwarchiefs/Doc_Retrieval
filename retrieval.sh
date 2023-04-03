output_dir=/cfs/cfs-i125txtf/jamsluo/lichee_retrieval/temp
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0
#init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/retriever_rocketqa_joint_4gpu_diag/checkpoint-8000/
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/classify_rank_top100.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
python3 run_retrival.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --do_train \
  --save_steps 4000 \
  --train_dir $train_data_dir \
  --q_max_len 24 \
  --p_max_len 192 \
  --logging_steps 100 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 5 \
  --gradient_accumulation_steps 1 \
  --fp16 \
  --adafactor \
  --train_group_size 5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy steps \
  --eval_steps 4000 \
  --pred_path $pred_path \
  --pred_id_file $pred_id_file
