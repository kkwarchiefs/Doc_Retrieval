conda activate env-3.6.8
pip install transformers==4.21.1
#pip install transformers==4.8.1
pip install datasets
cd /cfs/cfs-i125txtf/jamsluo/ft_local/apex-master
pip install -v --disable-pip-version-check --no-cache-dir ./
env CUDA_VISIBLE_DEVICES=0

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/tmp4
init_dir=/search/ai/pretrain_models/chinese-roberta-wwm-ext-large/
passage_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/passage_idx.pkl
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev.res.top100
python3 run_basic_du.py  \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --resume_from_checkpoint=/search/ai/jamsluo/passage_rank/du_task_output/tmp3/checkpoint-3000/ \
  --passage_path $passage_path \
  --logging_steps 50 \
  --do_train \
  --save_steps 1000 \
  --train_dir $train_data_dir \
  --max_len 128 \
  --seed 66 \
  --fp16 \
  --adafactor \
  --per_device_train_batch_size 2 \
  --train_group_size 3 \
  --per_device_eval_batch_size 64 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy steps \
  --eval_steps 4000 \
  --pred_path $pred_path

export CUDA_LAUNCH_BLOCKING=1
output_dir=/search/ai/jamsluo/passage_rank/du_task_output/roberta_large_g9_1e5
init_dir=/search/ai/pretrain_models/chinese-roberta-wwm-ext-large/
passage_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/passage_idx.pkl
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev.res.top4
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_du.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --passage_path $passage_path \
  --logging_steps 50 \
  --do_train \
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

export CUDA_LAUNCH_BLOCKING=1
output_dir=/search/ai/jamsluo/passage_rank/du_task_output/temp
init_dir=/search/ai/jamsluo/passage_rank/du_task_output/roberta_large_g9_1e5/checkpoint-9000/
passage_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/passage_idx.pkl
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev.res.top10
python3 -m torch.distributed.launch --nproc_per_node 4 retrival_du.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --passage_path $passage_path \
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

export CUDA_LAUNCH_BLOCKING=1
output_dir=/search/ai/jamsluo/passage_rank/du_task_output/ernie_base_g3_5e5_long
init_dir=/search/ai/pretrain_models/models--nghuyong--ernie-3.0-base-zh/
passage_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/passage_idx.pkl
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev.res.top10
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_du.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --passage_path $passage_path \
  --logging_steps 50 \
  --do_train \
  --save_steps 500 \
  --train_dir $train_data_dir \
  --q_max_len 64 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 30 \
  --train_group_size 3 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 30 \
  --overwrite_output_dir \
  --dataloader_num_workers 3 \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/infoxml_g3_5e5_64_512_four
init_dir=/search/ai/pretrain_models/infoxlm-base/
passage_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/passage_idx.pkl
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev.res.top10
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_du.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --passage_path $passage_path \
  --logging_steps 50 \
  --do_train \
  --save_steps 600 \
  --train_dir $train_data_dir \
  --q_max_len 64 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 30 \
  --train_group_size 3 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 30 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --pred_path $pred_path \
  --use_legacy_prediction_loop


output_dir=/search/ai/jamsluo/passage_rank/du_task_output/infoxml_g3_5e5_64_512_multiple_src
init_dir=/search/ai/pretrain_models/infoxlm-base/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_dual/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev.res.top10
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_du_mul.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_train \
  --save_steps 600 \
  --train_dir $train_data_dir \
  --q_max_len 64 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 30 \
  --train_group_size 3 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 30 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/infoxml_g2_5e5_128_512_squad
init_dir=/search/ai/pretrain_models/infoxlm-base/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair.tsv
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_squad.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_train \
  --save_steps 600 \
  --train_dir $train_data_dir \
  --q_max_len 128 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 30 \
  --train_group_size 2 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_multi_openqa_squad_pooling
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair.tsv
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_squad_pooling.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_train \
  --save_steps 1000 \
  --train_dir $train_data_dir \
  --q_max_len 128 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 44 \
  --train_group_size 2 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy steps \
  --save_steps 1000 \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_multi_openqa_squad_pooling_v2
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad_third/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair_msmarco5k_ducross5k.tsv
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_squad_pooling.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_train \
  --save_steps 1000 \
  --train_dir $train_data_dir \
  --q_max_len 128 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 44 \
  --train_group_size 2 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --pred_path $pred_path \
  --use_legacy_prediction_loop


output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_multi_openqa_squad_pooling_exact
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad_msmarco_exact/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair_msmarco_5k_exact.tsv
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_squad_pooling.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_train \
  --save_strategy epoch \
  --train_dir $train_data_dir \
  --q_max_len 128 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 44 \
  --train_group_size 2 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 4 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_m3e_openqa_squad_pooling_exact
init_dir=/search/ai/pretrain_models/m3e-base
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad_msmarco_exact/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair_msmarco_5k_exact.tsv
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_squad_pooling.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_train \
  --save_strategy epoch \
  --train_dir $train_data_dir \
  --q_max_len 128 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 44 \
  --train_group_size 2 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 4 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_multi_openqa_squad_pooling_exact_gpt
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad_msmarco_exact_gpt/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair_msmarco_5k_exact.tsv
torchrun --nproc_per_node=8 --master_port=8899 retrival_squad_pooling.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_train \
  --save_strategy epoch \
  --train_dir $train_data_dir \
  --q_max_len 128 \
  --p_max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 72 \
  --train_group_size 2 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 8 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --use_legacy_prediction_loop \
  --bf16 True \
  --fsdp "full_shard auto_wrap"

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



output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_multi_squad_colbert_dim64_exact_pdf8k_world
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad_msmarco_exact_pdf/
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
  --per_device_train_batch_size 22 \
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

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_m3e_squad_colbert_dim64_exact
init_dir=/search/ai/pretrain_models/m3e-base
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

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_multi_squad_colbert_dim64_exact_masking_v3
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad_msmarco_exact/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair_msmarco_5k_exact.tsv
env CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 6 retrival_squad_colbert.py \
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
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_multi_squad_colbert_dim64_exact_mktable
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad_msmarco_exact_table/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair_msmarco_5k_exact_2k_table.tsv
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
  --per_device_eval_batch_size 1 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_multi_squad_colbert_dim64_msmarco
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad_second/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair_msmarco_5k.tsv
python3 -m torch.distributed.launch --nproc_per_node 7 retrival_squad_colbert.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 10 \
  --do_train \
  --save_steps 200 \
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
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/passage_multi_squad_colbert_dim64_msmarco_dureader
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad_third/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair_msmarco5k_ducross5k.tsv
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_squad_colbert.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 10 \
  --do_train \
  --save_steps 1000 \
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
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 2 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --pred_path $pred_path \
  --use_legacy_prediction_loop





output_dir=/search/ai/jamsluo/passage_rank/du_task_output/rerank_mulit_mpnet_squad_v3
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_squad/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_squad_pair.tsv
python3 -m torch.distributed.launch --nproc_per_node 8 rerank_squad.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_train \
  --save_steps 500 \
  --train_dir $train_data_dir \
  --max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 12 \
  --train_group_size 8 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 4 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/rerank_mulit_mpnet_ms_du
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_dual/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_all.top10
python3 -m torch.distributed.launch --nproc_per_node 8 rerank_ms_du.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_train \
  --save_steps 500 \
  --train_dir $train_data_dir \
  --max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 9 \
  --train_group_size 16 \
  --per_device_eval_batch_size 128 \
  --fp16 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 4 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/rerank_mulit_mpnet_ms_du_baseline
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_dual_v2/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_all.top10
python3 -m torch.distributed.launch --nproc_per_node 8 rerank_ms_du.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_train \
  --save_steps 1000 \
  --train_dir $train_data_dir \
  --max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 9 \
  --train_group_size 16 \
  --per_device_eval_batch_size 128 \
  --fp16 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 4 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/rerank_mulit_mpnet_ms_du_baseline_point
init_dir=/search/ai/pretrain_models/paraphrase-multilingual-mpnet-base-v2/
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/train_dual_v2/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev_all.top10
python3 -m torch.distributed.launch --nproc_per_node 8 rerank_ms_du_point.py\
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --logging_steps 50 \
  --do_train \
  --save_steps 2000 \
  --train_dir $train_data_dir \
  --max_len 512 \
  --seed 66 \
  --per_device_train_batch_size 15 \
  --train_group_size 9 \
  --per_device_eval_batch_size 128 \
  --fp16 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 4 \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/search/ai/jamsluo/passage_rank/du_task_output/ernie_base_g2_5e5_dureader_train_sec
init_dir=/search/ai/pretrain_models/models--nghuyong--ernie-3.0-base-zh/
passage_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/passage_idx.pkl
train_data_dir=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/dureader-retrieval-baseline-dataset/train_dual/
pred_path=/search/ai/jamsluo/passage_rank/DuReader-Retrieval-Baseline/formate_data/dev/dev.res.top10
python3 -m torch.distributed.launch --nproc_per_node 8 retrival_du_line.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --passage_path $passage_path \
  --logging_steps 50 \
  --do_train \
  --save_steps 1000 \
  --train_dir $train_data_dir \
  --q_max_len 32 \
  --p_max_len 384 \
  --seed 66 \
  --per_device_train_batch_size 80 \
  --train_group_size 2 \
  --per_device_eval_batch_size 128 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 8 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --pred_path $pred_path \
  --use_legacy_prediction_loop

output_dir=/cfs/cfs-i125txtf/jamsluo/du_task_output/tmp
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/chinese-electra-180g-large-discriminator/
passage_path=/cfs/cfs-i125txtf/jamsluo/du_retrival_exp/DuReader-Retrieval-Baseline/formate_data/passage_idx.pkl
train_data_dir=/cfs/cfs-i125txtf/jamsluo/du_retrival_exp/DuReader-Retrieval-Baseline/formate_data/train/
pred_path=/cfs/cfs-i125txtf/jamsluo/du_retrival_exp/DuReader-Retrieval-Baseline/formate_data/dev/dev.res.top100
#python3 -m torch.distributed.launch --nproc_per_node 1  run_basic.py \
python3 run_basic_du.py  \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --passage_path $passage_path \
  --do_train \
  --save_steps 4000 \
  --train_dir $train_data_dir \
  --max_len 320 \
  --seed 66 \
  --fp16 \
  --adafactor \
  --per_device_train_batch_size 2 \
  --train_group_size 20 \
  --per_device_eval_batch_size 64 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --pred_path $pred_path

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
