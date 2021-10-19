#!/bin/bash

repo_last=$6

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



#output_dir=/datablob/users/v-sanych/s3prl_models/WavLM-Base
#
#python3 submit/submit.py \
#    --output_dir $output_dir \
#    --pr /datablob/users/v-sanych/s3prl_models/WavLM-Base_pr_bs128_lr5e-4_acc2 \
#    --sid /datablob/users/v-sanych/s3prl_models/WavLM-Base_sid_bs512_lr2e-1_acc1 \
#    --ks /datablob/users/v-sanych/s3prl_models/WavLM-Base_ks_bs512_lr1e-5_acc1 \
#    --ic /datablob/users/v-sanych/s3prl_models/WavLM-Base_ic_bs128_lr5e-5_acc1 \
#    --er_fold1 /datablob/users/v-sanych/s3prl_models/WavLM-Base_er_fold1_bs32_lr1e-4_acc1 \
#    --er_fold2 /datablob/users/v-sanych/s3prl_models/WavLM-Base_er_fold2_bs32_lr1e-4_acc1 \
#    --er_fold3 /datablob/users/v-sanych/s3prl_models/WavLM-Base_er_fold3_bs32_lr1e-4_acc1 \
#    --er_fold4 /datablob/users/v-sanych/s3prl_models/WavLM-Base_er_fold4_bs32_lr1e-4_acc1 \
#    --er_fold5 /datablob/users/v-sanych/s3prl_models/WavLM-Base_er_fold5_bs32_lr1e-4_acc1 \
#    --asr_no_lm /datablob/users/v-sanych/s3prl_models/WavLM-Base_asr_bs128_lr5e-4_acc1 \
#    --asr_with_lm /datablob/users/v-sanych/s3prl_models/WavLM-Base_asr_bs128_lr5e-4_acc1 \
#    --qbe /datablob/users/v-sanych/s3prl_models/WavLM-Base_qbe/hidden_state_12_test \
#    --sf /datablob/users/v-sanych/s3prl_models/WavLM-Base_sf_bs128_lr2e-4_acc1 \
#    --sv /datablob/users/v-sanych/s3prl_models/WavLM-Base_asv_bs512_lr5e-5_acc1 \
#    --sd /datablob/users/v-sanych/s3prl_models/WavLM-Base_sd_bs256_lr1e-3_acc4


#output_dir=/datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus
#
#python3 submit/submit.py \
#    --output_dir $output_dir \
#    --pr /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_pr_bs128_lr5e-4_acc2 \
#    --sid /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_sid_bs512_lr1e-1_acc1 \
#    --ks /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_ks_bs512_lr1e-3_acc1 \
#    --ic /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_ic_bs128_lr2e-5_acc1 \
#    --er_fold1 /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_er_fold1_bs32_lr1e-4_acc1 \
#    --er_fold2 /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_er_fold2_bs32_lr1e-4_acc1 \
#    --er_fold3 /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_er_fold3_bs32_lr1e-4_acc1 \
#    --er_fold4 /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_er_fold4_bs32_lr1e-4_acc1 \
#    --er_fold5 /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_er_fold5_bs32_lr1e-4_acc1 \
#    --asr_no_lm /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_asr_bs128_lr2e-4_acc1 \
#    --asr_with_lm /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_asr_bs128_lr2e-4_acc1 \
#    --qbe /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_qbe/hidden_state_12_test \
#    --sf /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_sf_bs128_lr2e-4_acc1 \
#    --sv /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_asv_bs512_lr5e-5_acc1 \
#    --sd /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus_sd_bs256_lr2e-3_acc4

#output_dir=/datablob/users/v-sanych/s3prl_models/WavLM-Large
#
#python3 submit/submit.py \
#    --output_dir $output_dir \
#    --pr /datablob/users/v-sanych/s3prl_models/WavLM-Large_pr_bs128_lr2e-4_acc2 \
#    --sid /datablob/users/v-sanych/s3prl_models/WavLM-Large_sid_bs512_lr1e-1_acc1 \
#    --ks /datablob/users/v-sanych/s3prl_models/WavLM-Large_ks_bs512_lr1e-6_acc1 \
#    --ic /datablob/users/v-sanych/s3prl_models/WavLM-Large_ic_bs128_lr2e-5_acc1 \
#    --er_fold1 /datablob/users/v-sanych/s3prl_models/WavLM-Large_er_fold1_bs32_lr1e-4_acc1 \
#    --er_fold2 /datablob/users/v-sanych/s3prl_models/WavLM-Large_er_fold2_bs32_lr1e-4_acc1 \
#    --er_fold3 /datablob/users/v-sanych/s3prl_models/WavLM-Large_er_fold3_bs32_lr1e-4_acc1 \
#    --er_fold4 /datablob/users/v-sanych/s3prl_models/WavLM-Large_er_fold4_bs32_lr1e-4_acc1 \
#    --er_fold5 /datablob/users/v-sanych/s3prl_models/WavLM-Large_er_fold5_bs32_lr1e-4_acc1 \
#    --asr_no_lm /datablob/users/v-sanych/s3prl_models/WavLM-Large_asr_bs128_lr2e-4_acc1 \
#    --asr_with_lm /datablob/users/v-sanych/s3prl_models/WavLM-Large_asr_bs128_lr2e-4_acc1 \
#    --qbe /datablob/users/v-sanych/s3prl_models/WavLM-Large_qbe/hidden_state_12_test \
#    --sf /datablob/users/v-sanych/s3prl_models/WavLM-Large_sf_bs128_lr2e-4_acc1 \
#    --sv /datablob/users/v-sanych/s3prl_models/WavLM-Large_asv_bs512_lr5e-5_acc1 \
#    --sd /datablob/users/v-sanych/s3prl_models/WavLM-Large_sd_bs256_lr2e-3_acc4

output_dir=/datablob/users/v-sanych/s3prl_models/WavLM-Large-More

python3 submit/submit.py \
    --output_dir $output_dir \
    --pr /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_pr_bs128_lr2e-4_acc2 \
    --sid /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_sid_bs512_lr5e-2_acc1 \
    --ks /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_ks_bs512_lr1e-5_acc1 \
    --ic /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_ic_bs128_lr5e-4_acc1 \
    --er_fold1 /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_er_fold1_bs32_lr1e-5_acc1 \
    --er_fold2 /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_er_fold2_bs32_lr1e-5_acc1 \
    --er_fold3 /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_er_fold3_bs32_lr1e-5_acc1 \
    --er_fold4 /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_er_fold4_bs32_lr1e-5_acc1 \
    --er_fold5 /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_er_fold5_bs32_lr1e-5_acc1 \
    --asr_no_lm /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_asr_bs128_lr1e-4_acc1 \
    --asr_with_lm /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_asr_bs128_lr1e-4_acc1 \
    --qbe /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_qbe/hidden_state_24_test \
    --sf /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_sf_bs128_lr1e-4_acc1 \
    --sv /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_asv_bs512_lr5e-5_acc1 \
    --sd /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_sd_bs256_lr5e-3_acc4