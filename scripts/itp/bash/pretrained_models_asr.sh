#!/bin/bash

repo_last=$6

sudo pip install torch_complex
#cp -r /datablob/users/v-sanych/sp_fairseq${repo_last} .
#cd sp_fairseq${repo_last}
cp -r /datablob/users/v-sanych/$1/$2/code .
cd code
echo "python setup.py install --user"
#pip install -e ./ --user
#python setup.py install --user
#pip install sentencepiece
sudo rm -r /fairseq
sudo pip install -e ./
sudo pip install sentencepiece
cd /tmp/code
sudo pip install -e ./
cd /tmp/code/s3prl
ls

model_dir=$1
model_name=$2
bs=$3
lr=$4
acc=$5

node=4
bs_per_node=$((bs / node / acc))

save_path=/datablob/users/v-sanych/s3prl_models/${model_name}_asr_bs${bs}_lr${lr}_acc${acc}_g8
model_path=/datablob/users/v-sanych/${model_dir}/${model_name}/checkpoint_last.pt

#sudo python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
#  -n ${model_name}_asr_bs${bs}_lr${lr}_acc${acc}  \
#  -p ${save_path}  \
#  -o "config.optimizer.lr=${lr},,config.downstream_expert.eval_batch_size=${bs_per_node}"  \
#  -c ./downstream/asr/config_bs${bs_per_node}_and_acc${acc}.yaml  \
#  -m train  \
#  -u hubert_local  \
#  -k ${model_path} \
#  -d asr  \
#  --verbose -a

ebs=$7
# python3 run_downstream.py -n ExpName -m train -u fbank -d asr

# Testing without LM
sudo python3 run_downstream.py -m evaluate -t "test-clean" -e ${save_path}/dev-clean-best.ckpt -o "config.downstream_expert.datarc.eval_batch_size=${ebs}" | tee ${save_path}/evaluate_results_wo_lm_submission.txt

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
    | tee ${save_path}/evaluate_results_w_lm_submission.txt

