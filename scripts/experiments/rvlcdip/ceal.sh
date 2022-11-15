#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_EXP_CONFIG_RVLCDIP="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_rvlcdip/resnet50"
BASE_EXP_CONFIG_RVLCDIP_IMB="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_rvlcdip/resnet50_imb"
BASE_EXP_CONFIG_RVLCDIP_N10="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_rvlcdip/resnet50_noisy_10"
BASE_EXP_CONFIG_RVLCDIP_N20="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_rvlcdip/resnet50_noisy_20"
CONFIGS=(
    "rvlcdip_resnet50_ceal_4k $BASE_EXP_CONFIG_RVLCDIP args/al_args=ceal args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=25 args.training_args.experiment_name=ceal_4k_fix" # 0
    "rvlcdip_resnet50_ceal_8k $BASE_EXP_CONFIG_RVLCDIP args/al_args=ceal args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=13 args.training_args.experiment_name=ceal_8k_fix" # 1
    # "rvlcdip_resnet50_ceal_16k $BASE_EXP_CONFIG_RVLCDIP args/al_args=ceal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=7 args.training_args.experiment_name=ceal_16k_fix" # 2
    "rvlcdip_resnet50_ceal_imb_2 $BASE_EXP_CONFIG_RVLCDIP_IMB args/al_args=ceal args.training_args.experiment_name=ceal_8k_imb_2_fix" # 3
    "rvlcdip_resnet50_ceal_imb_4 $BASE_EXP_CONFIG_RVLCDIP_IMB args/al_args=ceal args.al_args.n_classes_removed=4 args.training_args.experiment_name=ceal_8k_imb_4_fix" # 4
    "rvlcdip_resnet50_ceal_n10 $BASE_EXP_CONFIG_RVLCDIP_N10 args/al_args=ceal args.training_args.experiment_name=ceal_8k_n10_fix" # 4
    "rvlcdip_resnet50_ceal_n20 $BASE_EXP_CONFIG_RVLCDIP_N20 args/al_args=ceal args.training_args.experiment_name=ceal_8k_n20_fix" # 4
)