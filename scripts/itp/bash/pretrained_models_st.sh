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

# sp_fairseq
cp /tmp/code/s3prl/downstream/speech_translation/utils/init.py /tmp/code/code/fairseq/modules/__init__.py
cp /tmp/code/s3prl/downstream/speech_translation/utils/transformer_layer.py /tmp/code/code/fairseq/modules/transformer_layer.py
cp /tmp/code/s3prl/downstream/speech_translation/utils/ctc_prefix_score.py /tmp/code/code/fairseq/modules/ctc_prefix_score.py
cp /tmp/code/s3prl/downstream/speech_translation/utils/multihead_attention_fairseq.py /tmp/code/code/fairseq/modules/multihead_attention_fairseq.py

# new fairseq
#cp /tmp/code/s3prl/downstream/speech_translation/utils/speech_to_text_dataset.py /tmp/code/code/fairseq/data/audio/speech_to_text_dataset.py


save_path=/datablob/users/v-sanych/s3prl_models/${model_name}/st/bs${bs}_lr${lr}_acc${acc}_node${node}
model_path=/datablob/users/v-sanych/${model_dir}/${model_name}/checkpoint_last.pt

mkdir -p ${save_path}
python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
  -n ${model_name}/st/bs${bs}_lr${lr}_acc${acc}_node${node}  \
  -p ${save_path}  \
  -o "config.downstream_expert.datarc.max_tokens=${bs_per_node},,config.optimizer.lr=${lr},,config.runner.gradient_accumulate_steps=${acc}"  \
  -c ./downstream/speech_translation/config.yaml  \
  -m train  \
  -u hubert_local  \
  -k ${model_path} \
  -d speech_translation   \
  --verbose -a 2>&1 | tee -a ${save_path}/training_log.txt

python3 run_downstream.py -m evaluate -e ${save_path}/dev-best.ckpt -o "config.downstream_expert.datarc.max_tokens=10000" 2>&1 | tee -a ${save_path}/test_results.txt