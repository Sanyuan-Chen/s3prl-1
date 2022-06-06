#!/bin/bash

repo_last=$6

sudo apt-get install zip -y
#pip install torch_complex
##cp -r /datablob/users/v-sanych/sp_fairseq${repo_last} .
##cd sp_fairseq${repo_last}
#cp -r /datablob/users/v-sanych/$1/$2/code .
#cd code
#echo "python setup.py install --user"
##pip install -e ./ --user
#python setup.py install --user
#pip install sentencepiece
#cd /tmp/code
#pip install -e ./
cd /tmp/code/s3prl
ls

################## WavLM-Base ##################

#output_dir=/datablob/users/v-sanych/s3prl_models/WavLM-Base/`date "+%Y_%m_%d"`
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
#    --sd /datablob/users/v-sanych/s3prl_models/WavLM-Base/sd/bs256_lr2e-3_acc4_node8 \
#    --se /datablob/users/v-sanych/s3prl_models/WavLM-Base/se/bs64_lr5e-4_acc1_node4 \
#    --st /datablob/users/v-sanych/s3prl_models/WavLM-Base/st/bs80000_lr1e-3_acc1_node4 \
#    --ss /datablob/users/v-sanych/s3prl_models/WavLM-Base/ss/bs64_lr5e-4_acc1_node4

#    --sd /datablob/users/v-sanych/s3prl_models/WavLM-Base_sd_bs256_lr1e-3_acc4

################## WavLM-Base ##################

################## WavLM-Base-Plus ##################

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

################## WavLM-Base-Plus ##################

################## WavLM-Large ##################


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

################## WavLM-Large ##################


################## WavLM-Large-More ##################

#output_dir=/datablob/users/v-sanych/s3prl_models/WavLM-Large-More/`date "+%Y_%m_%d"`
#
#python3 submit/submit.py \
#    --output_dir $output_dir \
#    --pr /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_pr_bs128_lr2e-4_acc2 \
#    --sid /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_sid_bs512_lr5e-2_acc1 \
#    --ks /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_ks_bs512_lr1e-5_acc1 \
#    --ic /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_ic_bs128_lr5e-4_acc1 \
#    --er_fold1 /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_er_fold1_bs32_lr1e-5_acc1 \
#    --er_fold2 /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_er_fold2_bs32_lr1e-5_acc1 \
#    --er_fold3 /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_er_fold3_bs32_lr1e-5_acc1 \
#    --er_fold4 /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_er_fold4_bs32_lr1e-5_acc1 \
#    --er_fold5 /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_er_fold5_bs32_lr1e-5_acc1 \
#    --asr_no_lm /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_asr_bs128_lr1e-4_acc1 \
#    --asr_with_lm /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_asr_bs128_lr1e-4_acc1 \
#    --qbe /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_qbe/hidden_state_24_test \
#    --sf /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_sf_bs128_lr1e-4_acc1 \
#    --sv /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_asv_bs512_lr5e-5_acc1 \
#    --sd /datablob/users/v-sanych/s3prl_models/WavLM-Large-More/sd/bs256_lr1e-2_acc4_node8 \
#    --se /datablob/users/v-sanych/s3prl_models/WavLM-Large-More/se/bs64_lr5e-4_acc1_node8 \
#    --st /datablob/users/v-sanych/s3prl_models/WavLM-Large-More/st/bs160000_lr1e-3_acc2_node8 \
#    --ss /datablob/users/v-sanych/s3prl_models/WavLM-Large-More/ss/bs64_lr5e-4_acc1_node8

#    --sd /datablob/users/v-sanych/s3prl_models/WavLM-Large-More_sd_bs256_lr5e-3_acc4 \
################## WavLM-Large-More ##################


################## WavLM-Base-Plus-Noise ##################

#output_dir=/datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise/`date "+%Y_%m_%d"`
#
#python3 submit/submit.py \
#    --output_dir $output_dir \
#    --pr /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_pr_bs128_lr5e-4_acc2 \
#    --sid /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_sid_bs512_lr1e-1_acc1 \
#    --ks /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_ks_bs512_lr1e-5_acc1 \
#    --ic /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_ic_bs128_lr2e-5_acc1 \
#    --er_fold1 /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_er_fold1_bs32_lr1e-4_acc1 \
#    --er_fold2 /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_er_fold2_bs32_lr1e-4_acc1 \
#    --er_fold3 /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_er_fold3_bs32_lr1e-4_acc1 \
#    --er_fold4 /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_er_fold4_bs32_lr1e-4_acc1 \
#    --er_fold5 /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_er_fold5_bs32_lr1e-4_acc1 \
#    --asr_no_lm /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_asr_bs128_lr5e-4_acc1 \
#    --asr_with_lm /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_asr_bs128_lr5e-4_acc1 \
#    --qbe /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_qbe/hidden_state_12_test \
#    --sf /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_sf_bs128_lr2e-4_acc1 \
#    --sv /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise_asv_bs512_lr5e-5_acc1 \
#    --sd /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise/sd/bs256_lr5e-4_acc4_node8 \
#    --se /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise/se/bs64_lr5e-4_acc1_node4 \
#    --st /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise/st/bs80000_lr1e-3_acc1_node4 \
#    --ss /datablob/users/v-sanych/s3prl_models/WavLM-Base-Plus-Noise/ss/bs64_lr1e-3_acc1_node4

################## WavLM-Base-Plus-Noise ##################

################## WavLM-Large-Noise ##################
# mn_hubert_pretrain_large_94k_mnp01_mp02_grep_700k_475kstep_conti_700k_attrelax

output_dir=/datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/`date "+%Y_%m_%d"`

#
python3 submit/submit.py \
    --output_dir $output_dir \
    --pr /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/pr/bs128_lr2e-4_acc2_node8 \
    --sid /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/sid/bs512_lr5e-2_acc1_node8 \
    --ks /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/ks/bs512_lr1e-5_acc1_node4 \
    --ic /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/ic/bs128_lr5e-4_acc1_node4 \
    --er_fold1 /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/er/fold1_bs32_lr1e-5_acc1_node4 \
    --er_fold2 /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/er/fold2_bs32_lr1e-5_acc1_node4 \
    --er_fold3 /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/er/fold3_bs32_lr1e-5_acc1_node4 \
    --er_fold4 /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/er/fold4_bs32_lr1e-5_acc1_node4 \
    --er_fold5 /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/er/fold5_bs32_lr1e-5_acc1_node4 \
    --asr_no_lm /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/asr/bs128_lr1e-4_acc1_node8 \
    --asr_with_lm /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/asr/bs128_lr1e-4_acc1_node8 \
    --qbe /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/qbe/hidden_state_24_test \
    --sf /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/sf/bs128_lr1e-4_acc1_node8 \
    --sv /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/asv/bs512_lr5e-5_acc1_node8/states-130000 \
    --sd /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/sd/bs256_lr5e-3_acc4_node8 \
    --se /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/se/bs64_lr5e-4_acc1_node8 \
    --st /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/st/bs160000_lr1e-3_acc2_node8 \
    --ss /datablob/users/v-sanych/s3prl_models/WavLM-Large-Noise/ss/bs64_lr5e-4_acc1_node8

################## WavLM-Large-Noise ##################