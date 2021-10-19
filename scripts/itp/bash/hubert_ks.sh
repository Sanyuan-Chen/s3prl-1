#!/bin/bash


bs=$1
lr=$2
acc=$3

node=1
bs_per_node=$((bs / node / acc))

#save_path=/datablob/users/v-sanych/s3prl_models/hubert_ks_bs${bs}_lr1e${lr}_acc${acc}
#model_path=/datablob/users/v-sanych/pretrained_models/hubert_base_ls960.pt.fixed

save_path=/datablob/users/v-sanych/s3prl_models/hubert_large_ks_bs${bs}_lr1e${lr}_acc${acc}
model_path=/datablob/users/v-sanych/pretrained_models/hubert_large_ll60k.pt.fixed

python3 run_downstream.py  \
  -n hubert_ks_bs${bs}_lr1e${lr}_acc${acc}  \
  -p ${save_path}  \
  -o "config.optimizer.lr=1.0e-${lr}"  \
  -c ./downstream/speech_commands/config_bs${bs_per_node}_and_acc${acc}.yaml  \
  -m train  \
  -u hubert_local  \
  -k ${model_path} \
  -d speech_commands  \
  --verbose

# python3 run_downstream.py -n ExpName -m train -u fbank -d speech_commands

# Testing
python3 utility/get_best_dev.py ${save_path}/log.log