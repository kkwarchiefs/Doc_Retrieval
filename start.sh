conda activate env-3.6.8
pip install transformers==4.2.1
#pip install transformers==4.8.1
pip install datasets
cd /cfs/cfs-i125txtf/jamsluo/ft_local/apex-master
pip install -v --disable-pip-version-check --no-cache-dir ./
env CUDA_VISIBLE_DEVICES=0


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/ernie_basic_rocketqa2
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0
#init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/ernie_basic_rocketqa/checkpoint-108000
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
python3 -m torch.distributed.launch --nproc_per_node 1  run_basic.py \
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
  --per_device_train_batch_size 3 \
  --train_group_size 128 \
  --per_device_eval_batch_size 64 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 2e-6 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy steps \
  --eval_steps 4000 \
  --pred_path $pred_path \
  --pred_id_file $pred_id_file


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/lichee_inter_rocketqa_cls
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0/
#init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/lichee_inter_rocketqa/checkpoint-4000
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/classify_rank_top100.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
python3 -m torch.distributed.launch --nproc_per_node 4  run_interact.py \
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
  --pred_id_file $pred_id_file


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_longformer_rocketqa
#init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/longformer-base-4096
init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_longformer_rocketqa/checkpoint-18000
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/classify_rank_top100.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
python3 -m torch.distributed.launch --nproc_per_node 4  run_reclassify_longformer.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_train \
  --save_steps 6000 \
  --train_dir $train_data_dir \
  --max_len 128 \
  --seed 66 \
  --fp16 \
  --adafactor \
  --per_device_train_batch_size 6 \
  --train_group_size 30 \
  --per_device_eval_batch_size 8 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy steps \
  --eval_steps 6000 \
  --pred_path $pred_path \
  --pred_id_file $pred_id_file


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_intermlp_rocketqa
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0/
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/classify_rank_top100.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
python3 -m torch.distributed.launch --nproc_per_node 4  run_reclassify_mlp.py \
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
  --per_device_train_batch_size 20 \
  --train_group_size 20 \
  --per_device_eval_batch_size 32 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy steps \
  --eval_steps 4000 \
  --pred_path $pred_path \
  --pred_id_file $pred_id_file


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_att_rocketqa
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0/
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/classify_rank_top100.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
python3 -m torch.distributed.launch --nproc_per_node 4  run_reclassify_att.py \
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
  --per_device_train_batch_size 20 \
  --train_group_size 20 \
  --per_device_eval_batch_size 32 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy steps \
  --eval_steps 4000 \
  --pred_path $pred_path \
  --pred_id_file $pred_id_file


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_hinge_rocketqa
#init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0/
init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output//lichee_inter_hinge_rocketqa/checkpoint-112000
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/classify_rank_top100.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
python3 -m torch.distributed.launch --nproc_per_node 4  run_interact_hinge.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
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
  --pred_id_file $pred_id_file


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_cnn_rocketqa
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0/
#init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_cnn_rocketqa/checkpoint-8000
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/classify_rank_top100.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
python3 run_interact_cnn.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_train \
  --save_steps 4000 \
  --train_dir $train_data_dir \
  --max_len 128 \
  --seed 66 \
  --fp16 \
  --per_device_train_batch_size 4 \
  --train_group_size 100 \
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
  --fp16 \
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

output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/temp
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/bert-base-uncased/
#init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_pair_mlp_rocketqa/checkpoint-128000
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
  --fp16 \
  --per_device_train_batch_size 4 \
  --train_group_size 80 \
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
init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_pair_mlp_rocketqa/checkpoint-128000
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/classify_rank_top100.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/rank100.text.rank.tsv
rank_score_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_pair_mlp_rocketqa/res.tsv
python3 run_interact_pair_mlp.py \
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
  --rank_score_path $rank_score_path
