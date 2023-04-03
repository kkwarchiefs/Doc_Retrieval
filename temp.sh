output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/temp
#init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0/
init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_pair_mlp_rocketqa/checkpoint-128000
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/classify_rank_top100.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
python3 run_interact_pair_mlp.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --do_train \
  --save_steps 4000 \
  --train_dir $train_data_dir \
  --max_len 128 \
  --seed 66 \
  --per_device_train_batch_size 10 \
  --train_group_size 15 \
  --per_device_eval_batch_size 8 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy steps \
  --eval_steps 4000 \
  --pred_path $pred_path \
  --pred_id_file $pred_id_file


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/tmp
#init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0/
init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/lichee_pair_mlp_rocketqa_origin_diag_active
pred_path=/cfs/cfs-i125txtf/jamsluo/lichee_retrieval/outputs/coil_recall.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/lichee_retrieval/outputs/coil_recall.tsv
rank_score_path=/cfs/cfs-i125txtf/jamsluo/lichee_retrieval/outputs/coid_recall_res.tsv
python3  -m torch.distributed.launch --nproc_per_node 4 run_interact_pair_mlp.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --do_predict \
  --max_len 128 \
  --seed 66 \
  --train_group_size 15 \
  --per_device_eval_batch_size 20 \
  --overwrite_output_dir \
  --pred_path $pred_path \
  --pred_id_file $pred_id_file \
  --rank_score_path $rank_score_path
