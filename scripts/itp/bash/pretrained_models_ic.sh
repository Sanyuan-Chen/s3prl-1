#!/bin/bash

repo_last=$7

pip install torch_complex
#cp -r /datablob/users/v-sanych/sp_fairseq${repo_last} .
#cd sp_fairseq${repo_last}
cp -r /datablob/users/v-sanych/$1/$2/code .
cd code
echo "python setup.py install --user"
#pip install -e ./ --user
python setup.py install --user
pip install sentencepiece
cd /tmp/code
pip install -e ./
cd /tmp/code/s3prl
ls

model_dir=$1
model_name=$2
bs=$3
lr=$4
acc=$5
wu=$6

node=4
bs_per_node=$((bs / node / acc))

save_path=/datablob/users/v-sanych/s3prl_models/${model_name}_ic_bs${bs}_lr${lr}_acc${acc}
model_path=/datablob/users/v-sanych/${model_dir}/${model_name}/checkpoint_last.pt

#python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
#  -n ${model_name}_ic_bs${bs}_lr${lr}_acc${acc}  \
#  -p ${save_path}  \
#  -o "config.optimizer.lr=${lr}"  \
#  -c ./downstream/fluent_commands/config_bs${bs_per_node}_and_acc${acc}.yaml  \
#  -m train  \
#  -u hubert_local  \
#  -k ${model_path} \
#  -d fluent_commands  \
#  --verbose -a

#save_path=/datablob/users/v-sanych/s3prl_models/${model_name}_ic_bs${bs}_lr${lr}_acc${acc}_wu${wu}
#model_path=/datablob/users/v-sanych/${model_dir}/${model_name}/checkpoint_last.pt

#python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
#  -n ${model_name}_ic_bs${bs}_lr${lr}_acc${acc}_wu${wu}  \
#  -p ${save_path}  \
#  -o "config.optimizer.lr=${lr},,config.scheduler.num_warmup_steps=${wu}"  \
#  -c ./downstream/fluent_commands/config_bs${bs_per_node}_and_acc${acc}_wu.yaml  \
#  -m train  \
#  -u hubert_local  \
#  -k ${model_path} \
#  -d fluent_commands  \
#  --verbose -a

# python3 run_downstream.py -n ExpName -m train -u fbank -d fluent_commands

#python3 utility/get_best_dev.py ${save_path}/log.log

python3 run_downstream.py -m evaluate -e ${save_path}/dev-best.ckpt | tee ${save_path}/evaluate_results_submission.txt