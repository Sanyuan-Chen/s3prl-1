#!/bin/bash

model_dir=$1
model_name=$2
bs=$3
lr=$4
acc=$5
node=$6
bs_per_node=$((bs / node / acc))

pip install -e ./
pip install torch_complex
cp -r /datablob/users/v-sanych/${model_dir}/${model_name}/code .
cd code
sudo pip install --editable ./
cd /tmp/code/s3prl
ls

save_path=/datablob/users/v-sanych/s3prl_models/${model_name}/sid/bs${bs}_lr${lr}_acc${acc}_node${node}
model_path=/datablob/users/v-sanych/${model_dir}/${model_name}/checkpoint_last.pt

mkdir -p ${save_path}
python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
  -n ${model_name}/sid/bs${bs}_lr${lr}_acc${acc}_node${node}  \
  -p ${save_path}  \
  -o "config.downstream_expert.datarc.train_batch_size=${bs_per_node},,config.downstream_expert.datarc.eval_batch_size=1,,config.optimizer.lr=${lr},,config.runner.gradient_accumulate_steps=${acc}"  \
  -c ./downstream/voxceleb1/config.yaml  \
  -m train  \
  -u hubert_local  \
  -k ${model_path} \
  -d voxceleb1  \
  --verbose -a 2>&1 | tee -a ${save_path}/training_log.tx

python3 run_downstream.py -m evaluate -e ${save_path}/dev-best.ckpt 2>&1 | tee -a ${save_path}/evaluate_results.txt