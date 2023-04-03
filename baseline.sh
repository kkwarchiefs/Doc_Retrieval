conda activate env-3.6.8
pip install transformers==4.2.1
#pip install transformers==4.8.1
pip install datasets
cd /cfs/cfs-i125txtf/jamsluo/ft_local/apex-master
pip install -v --disable-pip-version-check --no-cache-dir ./
env CUDA_VISIBLE_DEVICES=0


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/tmp
#init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0/
init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_pair_mlp_rocketqa/checkpoint-128000
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/bm25_top1000_0519_v2.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/bm25_top1000_0519_v2.json
rank_score_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_pair_mlp_rocketqa/res_bm25_50_v2.tsv
env CUDA_VISIBLE_DEVICES=0 python3 run_interact_pair_mlp.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_predict \
  --max_len 128 \
  --seed 66 \
  --fp16 \
  --train_group_size 50 \
  --per_device_eval_batch_size 4 \
  --overwrite_output_dir \
  --pred_path $pred_path \
  --pred_id_file $pred_id_file \
  --rank_score_path $rank_score_path

output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/tmp
#init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/ernie-2.0/
init_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_pair_mlp_rocketqa/checkpoint-128000
train_data_dir=/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/rocketqa128/
pred_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/dev_rocket_0519.json
pred_id_file=/cfs/cfs-i125txtf/jamsluo/paper_task_output/dev_retrieval/dev_rocket_0519.json
rank_score_path=/cfs/cfs-i125txtf/jamsluo/paper_task_output/abtest_inter_pair_mlp_rocketqa/res_dev_rocket2.tsv
python3 run_interact_pair_mlp.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_predict \
  --max_len 128 \
  --seed 66 \
  --fp16 \
  --train_group_size 50 \
  --per_device_eval_batch_size 10 \
  --overwrite_output_dir \
  --pred_path $pred_path \
  --pred_id_file $pred_id_file \
  --rank_score_path $rank_score_path
