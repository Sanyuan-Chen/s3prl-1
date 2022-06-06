#!/bin/bash


bs=$1
lr=$2
acc=$3
fold=$4

node=4
bs_per_node=$((bs / node / acc))


#model_path=/datablob/users/v-sanych/pretrained_models/hubert_base_ls960.pt.fixed
#save_path=/datablob/users/v-sanych/s3prl_models/hubert_er_fold${fold}_bs${bs}_lr1e${lr}_acc${acc}

model_path=/datablob/users/v-sanych/pretrained_models/hubert_large_ll60k.pt.fixed
save_path=/datablob/users/v-sanych/s3prl_models/hubert_large_er_fold${fold}_bs${bs}_lr1e${lr}_acc${acc}

# The default config is "downstream/emotion/config.yaml"
python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py \
  -n hubert_er_fold${fold}_bs${bs}_lr1e${lr}_acc${acc}  \
  -p ${save_path}  \
  -o "config.optimizer.lr=1.0e-${lr},,config.downstream_expert.datarc.test_fold='fold${fold}'"  \
  --downstream_variant fold${fold} \
  -c ./downstream/emotion/config_bs${bs_per_node}_and_acc${acc}.yaml  \
  -m train  \
  -u hubert_local  \
  -k ${model_path} \
  -d emotion  \
  --verbose -a

#     python3 run_downstream.py  -n ExpName_$test_fold -m train -u fbank -d emotion -o "config.downstream_expert.datarc.test_fold='$test_fold'"

python3 utility/get_best_dev.py ${save_path}/log.log | tee ${save_path}/evaluate_results.txt

