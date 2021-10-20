#!/bin/bash

repo_last=$6

model_dir=$1
model_name=$2
bs=$3
lr=$4
acc=$5
node=$6
bs_per_node=$((bs / node / acc))

pip install torch_complex
cp -r /datablob/users/v-sanych/${model_dir}/${model_name}/code .
cd code
sudo pip install --editable ./
cd /tmp/code/s3prl
ls

save_path=/datablob/users/v-sanych/s3prl_models/${model_name}/asr/bs${bs}_lr${lr}_acc${acc}_node${node}
model_path=/datablob/users/v-sanych/${model_dir}/${model_name}/checkpoint_last.pt

sudo python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
  -n ${model_name}/asr/bs${bs}_lr${lr}_acc${acc}_node${node}  \
  -p ${save_path}  \
  -o "config.downstream_expert.loaderrc.train_batchsize=${bs_per_node},,config.downstream_expert.loaderrc.eval_batchsize=${bs_per_node},,config.optimizer.lr=${lr},,config.runner.gradient_accumulate_steps=${acc}"  \
  -c ./downstream/asr/config.yaml  \
  -m train  \
  -u hubert_local  \
  -k ${model_path} \
  -d asr  \
  --verbose -a

ebs=$7
# python3 run_downstream.py -n ExpName -m train -u fbank -d asr

# Testing without LM
sudo python3 run_downstream.py -m evaluate -t "test-clean" -e ${save_path}/dev-clean-best.ckpt -o "config.downstream_expert.datarc.eval_batch_size=${ebs}" | tee -a ${save_path}/evaluate_results_wo_lm.txt

# Testing with LM
sudo pip list
sudo python3 -c "from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder;print('okksuccess: from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder')"
sudo python3 -c "from flashlight.lib.text.decoder import CriterionType;print('okksuccess: from flashlight.lib.text.decoder import CriterionType')"
sudo python3 run_downstream.py -m evaluate -t "test-clean" -e ${save_path}/dev-clean-best.ckpt \
    -o "\
        config.downstream_expert.datarc.decoder_args.decoder_type='kenlm',, \
        config.downstream_expert.datarc.decoder_args.kenlm_model='/datablob/users/v-sanych/s3prl_data/ASR_TEST_UTILS/4-gram.arpa.gz',, \
        config.downstream_expert.datarc.decoder_args.lexicon='/datablob/users/v-sanych/s3prl_data/ASR_TEST_UTILS/librispeech_lexicon.lst' \
        ,,config.downstream_expert.datarc.eval_batch_size=${ebs} \
       " \
    | tee -a ${save_path}/evaluate_results_w_lm.txt

