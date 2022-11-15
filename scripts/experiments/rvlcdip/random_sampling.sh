#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_EXP_CONFIG_RVLCDIP="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50"
BASE_EXP_CONFIG_RVLCDIP_IMB="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50_imb"
BASE_EXP_CONFIG_RVLCDIP_N10="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50_noisy_10"
BASE_EXP_CONFIG_RVLCDIP_N20="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50_noisy_20"
CONFIGS=(
    # "rvlcdip_resnet50_random_sampling_4k $BASE_EXP_CONFIG_RVLCDIP args/al_args=random_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=25 args.training_args.experiment_name=random_sampling_4k" # 0
    "rvlcdip_resnet50_random_sampling_8k $BASE_EXP_CONFIG_RVLCDIP args/al_args=random_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=13 args.training_args.experiment_name=random_sampling_8k" # 1
    # "rvlcdip_resnet50_random_sampling_16k $BASE_EXP_CONFIG_RVLCDIP args/al_args=random_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=7 args.training_args.experiment_name=random_sampling_16k" # 2
    # "rvlcdip_resnet50_random_sampling_imb_2 $BASE_EXP_CONFIG_RVLCDIP_IMB args/al_args=random_sampling args.training_args.experiment_name=random_sampling_8k_imb_2" # 3
    # "rvlcdip_resnet50_random_sampling_imb_4 $BASE_EXP_CONFIG_RVLCDIP_IMB args/al_args=random_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=random_sampling_8k_imb_4" # 4
    # "rvlcdip_resnet50_random_sampling_n10 $BASE_EXP_CONFIG_RVLCDIP_N10 args/al_args=random_sampling args.training_args.experiment_name=random_sampling_8k_n10" # 4
    # "rvlcdip_resnet50_random_sampling_n20 $BASE_EXP_CONFIG_RVLCDIP_N20 args/al_args=random_sampling args.training_args.experiment_name=random_sampling_8k_n20" # 4
)
# $BASE_EXP_CONFIG_RVLCDIP args/al_args=random_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=13 args.training_args.experiment_name=random_sampling_8k_test
$BASE_EXP_CONFIG_RVLCDIP args/al_args=lpl args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=13 args.training_args.experiment_name=lpl_25_0_test2 args.al_args.al_seed=0 args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True