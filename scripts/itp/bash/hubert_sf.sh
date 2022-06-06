#!/bin/bash


bs=$1
lr=$2
acc=$3

node=4
bs_per_node=$((bs / node / acc))

#save_path=/datablob/users/v-sanych/s3prl_models/hubert_sf_bs${bs}_lr1e${lr}_acc${acc}
#model_path=/datablob/users/v-sanych/pretrained_models/hubert_base_ls960.pt.fixed

save_path=/datablob/users/v-sanych/s3prl_models/hubert_large_sf_bs${bs}_lr1e${lr}_acc${acc}
model_path=/datablob/users/v-sanych/pretrained_models/hubert_large_ll60k.pt.fixed

#python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
#  -n hubert_sf_bs${bs}_lr1e${lr}_acc${acc}  \
#  -p ${save_path}  \
#  -o "config.optimizer.lr=1.0e-${lr}"  \
#  -c ./downstream/ctc/snips_bs${bs_per_node}_and_acc${acc}.yaml  \
#  -m train  \
#  -u hubert_local  \
#  -k ${model_path} \
#  -d ctc  \
#  --verbose -a

# python3 run_downstream.py -n ExpName -m train -u fbank -d ctc -c downstream/ctc/snips.yaml

python3 run_downstream.py -m evaluate -e ${save_path}/valid-best.ckpt | tee ${save_path}/evaluate_results.txt