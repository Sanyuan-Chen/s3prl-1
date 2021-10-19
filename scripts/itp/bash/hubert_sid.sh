#!/bin/bash


bs=$1
lr=$2
acc=$3

node=8
bs_per_node=$((bs / node / acc))

#save_path=/datablob/users/v-sanych/s3prl_models/hubert_sid_bs${bs}_lr1e${lr}_acc${acc}
#model_path=/datablob/users/v-sanych/pretrained_models/hubert_base_ls960.pt.fixed

save_path=/datablob/users/v-sanych/s3prl_models/hubert_large_sid_bs${bs}_lr1e${lr}_acc${acc}
model_path=/datablob/users/v-sanych/pretrained_models/hubert_large_ll60k.pt.fixed

python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
  -n hubert_large_sid_bs${bs}_lr1e${lr}_acc${acc}  \
  -p ${save_path}  \
  -o "config.optimizer.lr=1.0e-${lr}"  \
  -c ./downstream/voxceleb1/config_bs${bs_per_node}_and_acc${acc}.yaml  \
  -m train  \
  -u hubert_local  \
  -k ${model_path} \
  -d voxceleb1  \
  --verbose -a

# python3 run_downstream.py -n ExpName -m train -u fbank -d fluent_commands

python3 utility/get_best_dev.py ${save_path}/log.log