#!/bin/bash

set -e

learning_rate=${1:-1e-3}
per_device_train_batch_size=${2:-512}
num_train_epochs=${3:-3}

root="."
data_root="${root}/data"
dataset="FB15K237"

model_name="TransE"
train_name="${model_name}-l${learning_rate}-b${per_device_train_batch_size}"
output_dir="${root}/checkpoints/${train_name}"

echo output_dir: $output_dir
mkdir -p $output_dir
timestamp=`date "+%m-%d-%H:%M"`
log_path="${output_dir}/${timestamp}.log"
echo log_path: $log_path

python -m torch_kge.run_train_kge \
  --data_dir $data_root \
  --dataset $dataset \
  --output_dir $output_dir \
  --do_train \
  --do_predict \
  --dim_ent 100 \
  --dim_rel 100 \
  --do_filter \
  --do_regularization \
  --margin 1 \
  --num_negs 2 \
  --neg_sample_strategy "uniform" \
  --logging_steps 100 \
  --per_device_train_batch_size $per_device_train_batch_size \
  --per_device_eval_batch_size 32 \
  --learning_rate $learning_rate \
  --num_train_epochs $num_train_epochs \
  --overwrite_output_dir \
  --weight_decay 0 \
  --lr_scheduler_type "constant" \
  --metric_for_best_model MRR \
  --save_total_limit 2 \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --eval_steps 100 \
  --load_best_model_at_end \
  --optim adamw_torch
