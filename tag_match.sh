source /search/ai/xiangyangli/anaconda3/bin/activate
conda activate torch131


output_dir=/search/ai/jamsluo/passage_rank/du_task_output/tag_match_m3e/
init_dir=/search/ai/pretrain_models/m3e-base/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/tag_concept_match/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/tag_match.tsv
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_squad_pooling.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 10 \
  --do_train \
  --save_strategy epoch \
  --train_dir $train_data_dir \
  --q_max_len 32 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 12 \
  --train_group_size 2 \
  --per_device_eval_batch_size 24 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --dataloader_num_workers 4 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --use_legacy_prediction_loop
