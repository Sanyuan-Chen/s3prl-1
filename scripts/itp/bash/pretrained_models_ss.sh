#!/bin/bash

model_dir=$1
model_name=$2
bs=$3
lr=$4
acc=$5
node=$6
bs_per_node=$((bs / node / acc))

win_length=$7
if [ _${win_length} = _ ];then
  win_length=512
fi

pip install -e ./
pip install torch_complex
cp -r /datablob/users/v-sanych/${model_dir}/${model_name}/code .
cd code
sudo pip install --editable ./
cd /tmp/code/s3prl
ls

save_path=/datablob/users/v-sanych/s3prl_models/${model_name}/ss/bs${bs}_lr${lr}_acc${acc}_node${node}_win_length${win_length}
model_path=/datablob/users/v-sanych/${model_dir}/${model_name}/checkpoint_last.pt

mkdir -p ${save_path}
python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
  -n ${model_name}/ss/bs${bs}_lr${lr}_acc${acc}_node${node}_win_length${win_length}  \
  -p ${save_path}  \
  -o "config.downstream_expert.loaderrc.train_batchsize=${bs_per_node},,config.downstream_expert.loaderrc.eval_batchsize=1,,config.optimizer.lr=${lr},,config.runner.gradient_accumulate_steps=${acc},,config.downstream_expert.datarc.n_fft=${win_length},,config.downstream_expert.datarc.win_length=${win_length}"  \
  -c ./downstream/separation_stft/configs/cfg.yaml  \
  -m train  \
  -u hubert_local  \
  -k ${model_path} \
  -d separation_stft    \
  --verbose -a 2>&1 | tee -a ${save_path}/training_log.txt

python3 run_downstream.py -m evaluate -e ${save_path}/best-states-dev.ckpt 2>&1 | tee -a ${save_path}/test_results.txt