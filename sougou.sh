conda activate env-3.6.8
pip install transformers==4.2.1
#pip install transformers==4.8.1
pip install datasets
cd /cfs/cfs-i125txtf/jamsluo/ft_local/apex-master
pip install -v --disable-pip-version-check --no-cache-dir ./
env CUDA_VISIBLE_DEVICES=0

270.0
214.0
299.0
322.0
472.0
729.0

output_dir=/cfs/cfs-i125txtf/jamsluo/du_task_output/tmp
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/chinese-roberta-wwm-ext/
#init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/ernie_basic_rocketqa/checkpoint-108000
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
  --max_len 384 \
  --seed 66 \
  --fp16 \
  --adafactor \
  --per_device_train_batch_size 2 \
  --train_group_size 40 \
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
