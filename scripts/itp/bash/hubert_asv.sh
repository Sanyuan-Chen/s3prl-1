#!/bin/bash


bs=$1
lr=$2
acc=$3

node=8
bs_per_node=$((bs / node / acc))

#save_path=/datablob/users/v-sanych/s3prl_models/hubert_sid_bs${bs}_lr1e${lr}_acc${acc}
#model_path=/datablob/users/v-sanych/pretrained_models/hubert_base_ls960.pt.fixed

save_path=/datablob/users/v-sanych/s3prl_models/hubert_large_asv_bs${bs}_lr1e${lr}_acc${acc}
model_path=/datablob/users/v-sanych/pretrained_models/hubert_large_ll60k.pt.fixed

#python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
#  -n hubert_large_asv_bs${bs}_lr1e${lr}_acc${acc}  \
#  -p ${save_path}  \
#  -o "config.optimizer.lr=1.0e-${lr}"  \
#  -c ./downstream/sv_voxceleb1/config_bs${bs_per_node}_and_acc${acc}.yaml  \
#  -m train  \
#  -u hubert_local  \
#  -k ${model_path} \
#  -d sv_voxceleb1  \
#  --verbose -a

./downstream/sv_voxceleb1/test_expdirs.sh ${save_path}
./downstream/sv_voxceleb1/report.sh ${save_path}