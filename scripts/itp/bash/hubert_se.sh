#!/bin/bash


bs=$1
lr=$2
acc=$3
node=$4
bs_per_node=$((bs / node / acc))

pip install -e ./
cd /tmp/code/s3prl
ls

#save_path=/datablob/users/v-sanych/s3prl_models/hubert/se/bs${bs}_lr${lr}_acc${acc}_node${node}
#model_path=/datablob/users/v-sanych/pretrained_models/hubert_base_ls960.pt.new

save_path=/datablob/users/v-sanych/s3prl_models/hubert_large/se/bs${bs}_lr${lr}_acc${acc}_node${node}
model_path=/datablob/users/v-sanych/pretrained_models/hubert_large_ll60k.pt

mkdir -p ${save_path}
python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
  -n hubert/se/bs${bs}_lr${lr}_acc${acc}_node${node}  \
  -p ${save_path}  \
  -o "config.downstream_expert.loaderrc.train_batchsize=${bs_per_node},,config.downstream_expert.loaderrc.eval_batchsize=1,,config.optimizer.lr=${lr},,config.runner.gradient_accumulate_steps=${acc}"  \
  -c ./downstream/enhancement_stft/configs/cfg_voicebank.yaml  \
  -m train  \
  -u hubert_local  \
  -k ${model_path} \
  -d enhancement_stft   \
  --verbose -a 2>&1 | tee -a ${save_path}/training_log.txt


python3 run_downstream.py -m evaluate -e ${save_path}/best-states-dev.ckpt 2>&1 | tee -a ${save_path}/test_results.txt