output_dir=/search/ai/jamsluo/passage_rank/du_task_output/temp
init_dir=/search/ai/jamsluo/passage_rank/du_task_output/infoxml_g3_5e5_64_512_multiple/checkpoint-6600/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_all.top10
python3 -m torch.distributed.launch --nproc_per_node 4 retrival_du_mul.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_eval \
  --save_steps 1000 \
  --train_dir $train_data_dir \
  --q_max_len 128 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 4 \
  --train_group_size 9 \
  --per_device_eval_batch_size 32 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --dataloader_num_workers 3 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --pred_path $pred_path \
  --use_legacy_prediction_loop
