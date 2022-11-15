#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_EXP_CONFIG_RVLCDIP="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_rvlcdip/resnet50"
BASE_EXP_CONFIG_RVLCDIP_IMB="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_rvlcdip/resnet50_imb"
BASE_EXP_CONFIG_RVLCDIP_N10="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_rvlcdip/resnet50_noisy_10"
BASE_EXP_CONFIG_RVLCDIP_N20="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_rvlcdip/resnet50_noisy_20"
CONFIGS=(
    "rvlcdip_resnet50_var_ratio_8k $BASE_EXP_CONFIG_RVLCDIP args/al_args=var_ratio args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=13 args.training_args.experiment_name=var_ratio_8k" # 1
)