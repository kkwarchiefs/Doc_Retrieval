output_dir=/cfs/cfs-i125txtf/jamsluo/event_workspace/event_rerank2/
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/chinese-roberta-wwm-ext/
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/event_dataset/train_dir/
pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/event_dataset/same_event_dev.json
python3 run_event_basic.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_train \
  --logging_steps 100 \
  --train_dir $train_data_dir \
  --save_steps 10000 \
  --max_len 96 \
  --seed 66 \
  --fp16 \
  --per_device_train_batch_size 10 \
  --train_group_size 7 \
  --per_device_eval_batch_size 10 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy epoch \
  --pred_path $pred_path
