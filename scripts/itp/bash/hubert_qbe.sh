#!/bin/bash
sudo apt-get update
sudo apt-get install bc default-jdk gnuplot -y
bc --version
java --version
gnuplot --version

model_path=/datablob/users/v-sanych/pretrained_models/hubert_base_ls960.pt.fixed
#model_path=/datablob/users/v-sanych/pretrained_models/hubert_large_ll60k.pt.fixed

for (( i = 0; i < 13; i++ )); do
#for (( i = 0; i < 25; i++ )); do

cd /tmp/code
# Dynamic Time Warping (DTW)

# The default dist_fn if not specified is "cosine_exp"
# as it yields the best result for almost all upstream
# Supported dist_fn: cosine, cityblock, euclidean, cosine_exp

dist_fn=cosine
feature_selection=hidden_state_${i} #[default, hidden_states]

save_path=/datablob/users/v-sanych/s3prl_models/hubert_qbe/${feature_selection}
#save_path=/datablob/users/v-sanych/s3prl_models/hubert_large_qbe/${feature_selection}

# dev
python3 run_downstream.py  \
  -n hubert_qbe_dev  \
  -p ${save_path}_dev  \
  -o "config.downstream_expert.dtwrc.dist_method='$dist_fn'"  \
  -c ./downstream/quesst14_dtw/config.yaml  \
  -m evaluate  \
  -u hubert_local  \
  -k ${model_path} \
  -d quesst14_dtw  \
  -t "dev" \
  -s ${feature_selection} \
  --verbose


# test
python3 run_downstream.py  \
  -n hubert_qbe_test  \
  -p ${save_path}_test  \
  -o "config.downstream_expert.dtwrc.dist_method='$dist_fn'"  \
  -c ./downstream/quesst14_dtw/config.yaml  \
  -m evaluate  \
  -u hubert_local  \
  -k ${model_path} \
  -d quesst14_dtw  \
  -t "test" \
  -s ${feature_selection} \
  --verbose


# Scoring

cd /datablob/users/v-sanych/s3prl_data/quesst14Database/scoring

# dev
./score-TWV-Cnxe.sh ${save_path}_dev \
    groundtruth_quesst14_dev -10

# test
./score-TWV-Cnxe.sh ${save_path}_test \
    groundtruth_quesst14_eval -10


done

