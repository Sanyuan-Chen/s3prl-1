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

save_path=/datablob/users/v-sanych/s3prl_models/${model_name}/asv/bs${bs}_lr${lr}_acc${acc}_node${node}
model_path=/datablob/users/v-sanych/${model_dir}/${model_name}/checkpoint_last.pt

mkdir -p ${save_path}
python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
  -n ${model_name}/asv/bs${bs}_lr${lr}_acc${acc}_node${node}  \
  -p ${save_path}  \
  -o "config.downstream_expert.loaderrc.train_batchsize=${bs_per_node},,config.downstream_expert.loaderrc.eval_batchsize=${bs_per_node},,config.optimizer.lr=${lr},,config.runner.gradient_accumulate_steps=${acc}"  \
  -c ./downstream/sv_voxceleb1/config.yaml  \
  -m train  \
  -u hubert_local  \
  -k ${model_path} \
  -d sv_voxceleb1  \
  --verbose -a 2>&1 | tee -a ${save_path}/training_log.txt

voxceleb1=/datablob/users/v-sanych/s3prl_data/VoxCeleb1
./downstream/sv_voxceleb1/test_expdir.sh ${save_path} $voxceleb1

#ckpt=$7
#bs=$8
#python3 run_downstream.py -m evaluate -e ${save_path}/states-${ckpt}.ckpt -o "config.downstream_expert.datarc.eval_batch_size=${bs}" | tee ${save_path}/evaluate_results_${ckpt}.txt