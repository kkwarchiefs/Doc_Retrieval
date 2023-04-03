output_dir=/cfs/cfs-i125txtf/jamsluo/event_workspace/event_retrieval_cosine/
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/chinese-roberta-wwm-ext/
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/event_dataset/train_dir/
pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/event_dataset/same_event_dev.json
env CUDA_VISIBLE_DEVICES=0 python3 run_event_retrival.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --do_train \
  --train_dir $train_data_dir \
  --q_max_len 48 \
  --p_max_len 48 \
  --logging_steps 100 \
  --save_steps 10000 \
  --per_device_train_batch_size 10 \
  --per_device_eval_batch_size 10 \
  --train_group_size 7 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --evaluation_strategy epoch \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --pred_path $pred_path
