#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_EXP_CONFIG_rvlcdip="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50"
BASE_EXP_CONFIG_rvlcdip_IMB="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50_imb"
BASE_EXP_CONFIG_rvlcdip_N10="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50_noisy_10"
BASE_EXP_CONFIG_rvlcdip_N20="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50_noisy_20"
CONFIGS=(
    # "rvlcdip_resnet50_waal_125 $BASE_EXP_CONFIG_rvlcdip args/al_args=waal args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=25 args.training_args.experiment_name=waal_125_0 args.al_args.al_seed=0 args/training_args=waal args.model_args.name=resnet50_waal" # 0
    "rvlcdip_resnet50_waal_25 $BASE_EXP_CONFIG_rvlcdip args/al_args=waal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=7 args.training_args.experiment_name=waal_25_0 args.al_args.al_seed=0 args/training_args=waal args.model_args.name=resnet50_waal" # 1
    # "rvlcdip_resnet50_waal_5 $BASE_EXP_CONFIG_rvlcdip args/al_args=waal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=7 args.training_args.experiment_name=waal_5_0 args.al_args.al_seed=0 args/training_args=waal args.model_args.name=resnet50_waal" # 2
    "rvlcdip_resnet50_waal_imb_2 $BASE_EXP_CONFIG_rvlcdip_IMB args/al_args=waal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=7 args.training_args.experiment_name=waal_25_imb_2_0 args.al_args.al_seed=0 args/training_args=waal args.model_args.name=resnet50_waal" # 3
    "rvlcdip_resnet50_waal_imb_4 $BASE_EXP_CONFIG_rvlcdip_IMB args/al_args=waal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=7 args.al_args.n_classes_removed=4 args.training_args.experiment_name=waal_25_imb_4_0 args.al_args.al_seed=0 args/training_args=waal args.model_args.name=resnet50_waal" # 4
    "rvlcdip_resnet50_waal_n10 $BASE_EXP_CONFIG_rvlcdip_N10 args/al_args=waal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=7 args.training_args.experiment_name=waal_25_n10_0 args.al_args.al_seed=0 args/training_args=waal args.model_args.name=resnet50_waal" # 4
    "rvlcdip_resnet50_waal_n20 $BASE_EXP_CONFIG_rvlcdip_N20 args/al_args=waal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=7 args.training_args.experiment_name=waal_25_n20_0 args.al_args.al_seed=0 args/training_args=waal args.model_args.name=resnet50_waal" #
)