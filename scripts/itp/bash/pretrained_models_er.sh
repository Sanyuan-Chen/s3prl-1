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
fold=$6

node=4
bs_per_node=$((bs / node / acc))

save_path=/datablob/users/v-sanych/s3prl_models/${model_name}_er_fold${fold}_bs${bs}_lr${lr}_acc${acc}
model_path=/datablob/users/v-sanych/${model_dir}/${model_name}/checkpoint_last.pt

## The default config is "downstream/emotion/config.yaml"
#python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py \
#  -n ${model_name}_er_fold${fold}_bs${bs}_lr${lr}_acc${acc}  \
#  -p ${save_path}  \
#  -o "config.optimizer.lr=${lr},,config.downstream_expert.datarc.test_fold='fold${fold}'"  \
#  --downstream_variant fold${fold} \
#  -c ./downstream/emotion/config_bs${bs_per_node}_and_acc${acc}.yaml  \
#  -m train  \
#  -u hubert_local  \
#  -k ${model_path} \
#  -d emotion  \
#  --verbose -a

#     python3 run_downstream.py  -n ExpName_$test_fold -m train -u fbank -d emotion -o "config.downstream_expert.datarc.test_fold='$test_fold'"

#for (( i = 1; i < 6; i++ )); do
#
#model_path=/datablob/users/v-sanych/s3prl_models/${model_name}_er_fold${i}_bs${bs}_lr${lr}_acc${acc}
#save_path=/datablob/users/v-sanych/s3prl_models/${model_name}_er_fold1_bs${bs}_lr${lr}_acc${acc}
#echo model_path | tee -a ${save_path}/evaluate_results.txt
#python3 utility/get_best_dev.py ${model_path}/log.log | tee -a ${save_path}/evaluate_results.txt
#
#done

for (( i = 1; i < 6; i++ )); do

save_path=/datablob/users/v-sanych/s3prl_models/${model_name}_er_fold${i}_bs${bs}_lr${lr}_acc${acc}
echo model_path | tee -a ${save_path}/evaluate_results.txt

python3 run_downstream.py -m evaluate -e ${save_path}/dev-best.ckpt  | tee ${save_path}/test_results_submission.txt

done