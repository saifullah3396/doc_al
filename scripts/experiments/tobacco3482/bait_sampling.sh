#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_EXP_CONFIG_TOBACCO3482="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_tobacco/resnet50"
BASE_EXP_CONFIG_TOBACCO3482_IMB="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_tobacco/resnet50_imb"
BASE_EXP_CONFIG_TOBACCO3482_N10="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_tobacco/resnet50_noisy_10"
BASE_EXP_CONFIG_TOBACCO3482_N20="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_tobacco/resnet50_noisy_20"
CONFIGS=(
    "tobacco3482_resnet50_bait_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=bait_sampling_125 args.al_args.al_seed=0" # 0
    "tobacco3482_resnet50_bait_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=bait_sampling_25 args.al_args.al_seed=0" # 1
    "tobacco3482_resnet50_bait_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=bait_sampling_5 args.al_args.al_seed=0" # 2
    "tobacco3482_resnet50_bait_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_imb_2 args.al_args.al_seed=0" # 3
    "tobacco3482_resnet50_bait_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=bait_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=bait_sampling_25_imb_4 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_bait_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_n10 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_bait_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_n20 args.al_args.al_seed=0" #

    "tobacco3482_resnet50_bait_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=bait_sampling_125_1 args.al_args.al_seed=1" # 0
    "tobacco3482_resnet50_bait_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=bait_sampling_25_1 args.al_args.al_seed=1" # 1
    "tobacco3482_resnet50_bait_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=bait_sampling_5_1 args.al_args.al_seed=1" # 2
    "tobacco3482_resnet50_bait_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_imb_2_1 args.al_args.al_seed=1" # 3
    "tobacco3482_resnet50_bait_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=bait_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=bait_sampling_25_imb_4_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_bait_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_n10_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_bait_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_n20_1 args.al_args.al_seed=1" #

    "tobacco3482_resnet50_bait_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=bait_sampling_125_2 args.al_args.al_seed=2" # 0
    "tobacco3482_resnet50_bait_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=bait_sampling_25_2 args.al_args.al_seed=2" # 1
    "tobacco3482_resnet50_bait_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=bait_sampling_5_2 args.al_args.al_seed=2" # 2
    "tobacco3482_resnet50_bait_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_imb_2_2 args.al_args.al_seed=2" # 3
    "tobacco3482_resnet50_bait_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=bait_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=bait_sampling_25_imb_4_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_bait_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_n10_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_bait_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_n20_2 args.al_args.al_seed=2" #

    "tobacco3482_resnet50_bait_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=bait_sampling_125_3 args.al_args.al_seed=3" # 0
    "tobacco3482_resnet50_bait_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=bait_sampling_25_3 args.al_args.al_seed=3" # 1
    "tobacco3482_resnet50_bait_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=bait_sampling_5_3 args.al_args.al_seed=3" # 2
    "tobacco3482_resnet50_bait_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_imb_2_3 args.al_args.al_seed=3" # 3
    "tobacco3482_resnet50_bait_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=bait_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=bait_sampling_25_imb_4_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_bait_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_n10_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_bait_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_n20_3 args.al_args.al_seed=3" #

    "tobacco3482_resnet50_bait_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=bait_sampling_125_4 args.al_args.al_seed=4" # 0
    "tobacco3482_resnet50_bait_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=bait_sampling_25_4 args.al_args.al_seed=4" # 1
    "tobacco3482_resnet50_bait_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bait_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=bait_sampling_5_4 args.al_args.al_seed=4" # 2
    "tobacco3482_resnet50_bait_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_imb_2_4 args.al_args.al_seed=4" # 3
    "tobacco3482_resnet50_bait_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=bait_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=bait_sampling_25_imb_4_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_bait_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_n10_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_bait_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=bait_sampling args.training_args.experiment_name=bait_sampling_25_n20_4 args.al_args.al_seed=4" # 4
)
