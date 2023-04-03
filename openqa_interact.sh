conda activate env-3.6.8
pip install transformers==4.2.1
#pip install transformers==4.8.1
pip install datasets
cd /cfs/cfs-i125txtf/jamsluo/ft_local/apex-master
pip install -v --disable-pip-version-check --no-cache-dir ./

output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/openqa_trecr_roberta_interact_vid
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/roberta-base
train_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/TRECR/trecr_train.json
pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/TRECR/trecr_valid.json
#pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/TRECR/trecr_test.json
python3 run_openqa_interact.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_train \
  --train_path $train_path \
  --max_len 256 \
  --seed 66 \
  --per_device_train_batch_size 6 \
  --train_group_size 20 \
  --eval_group_size 112 \
  --per_device_eval_batch_size 12 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 2e-5 \
  --logging_steps 100 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --pred_id_file $pred_path

output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/openqa_trecc_roberta_interact2
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/roberta-base
train_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/TRECC/trecc_train.json
#pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/TRECC/trecc_valid.json
pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/TRECC/trecc_test.json
python3 run_openqa_interact.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_train \
  --train_path $train_path \
  --max_len 128 \
  --seed 66 \
  --per_device_train_batch_size 10 \
  --train_group_size 20 \
  --eval_group_size 112 \
  --per_device_eval_batch_size 12 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 2e-5 \
  --logging_steps 100 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --pred_id_file $pred_path


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/openqa_wiki_roberta_interact2
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/roberta-base
train_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/WIKI/wiki_train.json
#pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/WIKI/wiki_valid.json
pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/WIKI/wiki_test.json
python3 run_openqa_interact.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_train \
  --train_path $train_path \
  --max_len 128 \
  --seed 66 \
  --per_device_train_batch_size 10 \
  --train_group_size 30 \
  --eval_group_size 112 \
  --per_device_eval_batch_size 12 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 2e-5 \
  --logging_steps 100 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --pred_id_file $pred_path


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/openqa_2015_roberta_interact2
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/roberta-base
train_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/SEMEVAL2015/semeval2015_train.json
#pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/SEMEVAL2015/semeval2015_valid.json
pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/SEMEVAL2015/semeval2015_test.json
python3 run_openqa_interact.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_train \
  --train_path $train_path \
  --max_len 128 \
  --seed 66 \
  --per_device_train_batch_size 10 \
  --train_group_size 25 \
  --eval_group_size 112 \
  --per_device_eval_batch_size 12 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 2e-5 \
  --logging_steps 100 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --pred_id_file $pred_path

output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/openqa_2016_roberta_interact_vid2
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/roberta-base
train_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/SEMEVAL2016/semeval2016_train.json
#pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/SEMEVAL2016/semeval2016_valid.json
pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/SEMEVAL2016/semeval2016_test.json
python3 run_openqa_interact.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_train \
  --train_path $train_path \
  --max_len 128 \
  --seed 66 \
  --per_device_train_batch_size 10 \
  --train_group_size 3 \
  --eval_group_size 112 \
  --per_device_eval_batch_size 12 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 2e-5 \
  --logging_steps 100 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --pred_id_file $pred_path


output_dir=/cfs/cfs-i125txtf/jamsluo/paper_task_output/openqa_2017_roberta_interact_vid2
init_dir=/cfs/cfs-i125txtf/jamsluo/dataset/pretrained_models/roberta-base
train_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/SEMEVAL2017/semeval2017_train.json
#pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/SEMEVAL2017/semeval2017_valid.json
pred_path=/cfs/cfs-i125txtf/jamsluo/dataset/OpenQA/openqa/SEMEVAL2017/semeval2017_test.json
python3 run_openqa_interact.py \
  --output_dir $output_dir \
  --model_name_or_path  $init_dir \
  --reload_path $init_dir \
  --do_train \
  --train_path $train_path \
  --max_len 256 \
  --seed 66 \
  --per_device_train_batch_size 10 \
  --train_group_size 4 \
  --eval_group_size 112 \
  --per_device_eval_batch_size 12 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 2e-5 \
  --logging_steps 100 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --evaluation_strategy epoch \
  --pred_path $pred_path \
  --pred_id_file $pred_path
