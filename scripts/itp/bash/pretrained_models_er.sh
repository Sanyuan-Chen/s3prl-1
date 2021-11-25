#!/bin/bash

model_dir=$1
model_name=$2
bs=$3
lr=$4
acc=$5
node=$6
fold=$7
bs_per_node=$((bs / node / acc))

pip install -e ./
pip install torch_complex
cp -r /datablob/users/v-sanych/${model_dir}/${model_name}/code .
cd code
sudo pip install --editable ./
cd /tmp/code/s3prl
ls

save_path=/datablob/users/v-sanych/s3prl_models/${model_name}/er/fold${fold}_bs${bs}_lr${lr}_acc${acc}_node${node}
model_path=/datablob/users/v-sanych/${model_dir}/${model_name}/checkpoint_last.pt

mkdir -p ${save_path}
## The default config is "downstream/emotion/config.yaml"
python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py \
  -n ${model_name}/er/fold${fold}_bs${bs}_lr${lr}_acc${acc}_node${node}  \
  -p ${save_path}  \
  -o "config.downstream_expert.datarc.train_batch_size=${bs_per_node},,config.downstream_expert.datarc.eval_batch_size=${bs_per_node},,config.optimizer.lr=${lr},,config.runner.gradient_accumulate_steps=${acc},,config.downstream_expert.datarc.test_fold='fold${fold}'"  \
  --downstream_variant fold${fold} \
  -c ./downstream/emotion/config.yaml  \
  -m train  \
  -u hubert_local  \
  -k ${model_path} \
  -d emotion  \
  --verbose -a 2>&1 | tee -a ${save_path}/training_log.txt

for (( i = 1; i < 6; i++ )); do

save_path=/datablob/users/v-sanych/s3prl_models/${model_name}/er/fold${i}_bs${bs}_lr${lr}_acc${acc}_node${node}
echo model_path | tee -a ${save_path}/evaluate_results.txt

python3 run_downstream.py -m evaluate -e ${save_path}/dev-best.ckpt  2>&1 | tee -a ${save_path}/evaluate_results.txt

done