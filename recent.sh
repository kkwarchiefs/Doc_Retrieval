output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_multi_squad_colbert_dim64_exact
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad_msmarco_exact/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair_msmarco_5k_exact.tsv
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_squad_colbert.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 10 \
  --do_train \
  --save_strategy epoch \
  --train_dir $train_data_dir \
  --q_max_len 128 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 48 \
  --train_group_size 2 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --use_legacy_prediction_loop
