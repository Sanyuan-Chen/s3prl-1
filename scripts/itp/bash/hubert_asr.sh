#!/bin/bash


bs=$1
lr=$2
acc=$3

node=4
bs_per_node=$((bs / node / acc))

#save_path=/datablob/users/v-sanych/s3prl_models/hubert_asr_bs${bs}_lr1e${lr}_acc${acc}
#model_path=/datablob/users/v-sanych/pretrained_models/hubert_base_ls960.pt.fixed

save_path=/datablob/users/v-sanych/s3prl_models/hubert_large_asr_bs${bs}_lr1e${lr}_acc${acc}
model_path=/datablob/users/v-sanych/pretrained_models/hubert_large_ll60k.pt.fixed

#python3 -m torch.distributed.launch --nproc_per_node ${node} run_downstream.py  \
#  -n hubert_asr_bs${bs}_lr1e${lr}_acc${acc}  \
#  -p ${save_path}  \
#  -o "config.optimizer.lr=1.0e-${lr}"  \
#  -c ./downstream/asr/config_bs${bs_per_node}_and_acc${acc}.yaml  \
#  -m train  \
#  -u hubert_local  \
#  -k ${model_path} \
#  -d asr  \
#  --verbose -a

# python3 run_downstream.py -n ExpName -m train -u fbank -d asr

# Testing without LM
python3 run_downstream.py -m evaluate -t "test-clean" -e ${save_path}/dev-clean-best.ckpt | tee ${save_path}/evaluate_results_wo_lm.txt

# Testing with LM
pip list
sudo python3 -c "from flashlight.lib.text.decoder import CriterionType;print('okksuccess')"
sudo python3 run_downstream.py -m evaluate -t "test-clean" -e ${save_path}/dev-clean-best.ckpt \
    -o "\
        config.downstream_expert.datarc.decoder_args.decoder_type='kenlm',, \
        config.downstream_expert.datarc.decoder_args.kenlm_model='/datablob/users/v-sanych/s3prl_data/ASR_TEST_UTILS/4-gram.arpa.gz',, \
        config.downstream_expert.datarc.decoder_args.lexicon='/datablob/users/v-sanych/s3prl_data/ASR_TEST_UTILS/librispeech_lexicon.lst' \
       " \
    | tee ${save_path}/evaluate_results_w_lm.txt