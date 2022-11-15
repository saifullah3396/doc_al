#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_EXP_CONFIG_TOBACCO3482="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_tobacco_pre/resnet50"
BASE_EXP_CONFIG_TOBACCO3482_IMB="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_tobacco_pre/resnet50_imb"
BASE_EXP_CONFIG_TOBACCO3482_N10="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_tobacco_pre/resnet50_noisy_10"
BASE_EXP_CONFIG_TOBACCO3482_N20="$SCRIPT_DIR/../../../train.sh --config-path ../../../cfg/al_32k/ +experiment=active_learning/new_tobacco_pre/resnet50_noisy_20"
CONFIGS=(
    # "tobacco3482_resnet50_bald_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bald_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=bald_dropout_25_1 args.al_args.al_seed=1" # 1

    # "tobacco3482_resnet50_ceal_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=ceal_125_1 args.al_args.al_seed=1" # 0
    # "tobacco3482_resnet50_ceal_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=ceal_25_1 args.al_args.al_seed=1" # 1
    # "tobacco3482_resnet50_ceal_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=ceal_5_1 args.al_args.al_seed=1" # 2
    # "tobacco3482_resnet50_ceal_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=ceal args.training_args.experiment_name=ceal_25_imb_2_1 args.al_args.al_seed=1" # 3
    # "tobacco3482_resnet50_ceal_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=ceal args.al_args.n_classes_removed=4 args.training_args.experiment_name=ceal_25_imb_4_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_ceal_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=ceal args.training_args.experiment_name=ceal_25_n10_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_ceal_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=ceal args.training_args.experiment_name=ceal_25_n20_1 args.al_args.al_seed=1" # 4

    # "tobacco3482_resnet50_entropy_sampling_dropout_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=entropy_sampling_dropout_125_1 args.al_args.al_seed=1" # 0
    # "tobacco3482_resnet50_entropy_sampling_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=entropy_sampling_dropout_25_1 args.al_args.al_seed=1" # 1
    # "tobacco3482_resnet50_entropy_sampling_dropout_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=entropy_sampling_dropout_5_1 args.al_args.al_seed=1" # 2
    # "tobacco3482_resnet50_entropy_sampling_dropout_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_imb_2_1 args.al_args.al_seed=1" # 3
    # "tobacco3482_resnet50_entropy_sampling_dropout_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=entropy_sampling_dropout_25_imb_4_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_entropy_sampling_dropout_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_n10_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_entropy_sampling_dropout_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_n20_1 args.al_args.al_seed=1" # 4

    # "tobacco3482_resnet50_entropy_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=entropy_sampling_125_1 args.al_args.al_seed=1" # 0
    # "tobacco3482_resnet50_entropy_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=entropy_sampling_25_1 args.al_args.al_seed=1" # 1
    # "tobacco3482_resnet50_entropy_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=entropy_sampling_5_1 args.al_args.al_seed=1" # 2
    # "tobacco3482_resnet50_entropy_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_imb_2_1 args.al_args.al_seed=1" # 3
    # "tobacco3482_resnet50_entropy_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=entropy_sampling_25_imb_4_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_entropy_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_n10_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_entropy_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_n20_1 args.al_args.al_seed=1" # 4

    # "tobacco3482_resnet50_kcenter_greedy_safe_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=kcenter_greedy_safe_125_1 args.al_args.al_seed=1" # 0
    # "tobacco3482_resnet50_kcenter_greedy_safe_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=kcenter_greedy_safe_25_1 args.al_args.al_seed=1" # 1
    # "tobacco3482_resnet50_kcenter_greedy_safe_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=kcenter_greedy_safe_5_1 args.al_args.al_seed=1" # 2
    # "tobacco3482_resnet50_kcenter_greedy_safe_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_imb_2_1 args.al_args.al_seed=1" # 3
    # "tobacco3482_resnet50_kcenter_greedy_safe_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=kcenter_greedy_safe args.al_args.n_classes_removed=4 args.training_args.experiment_name=kcenter_greedy_safe_25_imb_4_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_kcenter_greedy_safe_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_n10_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_kcenter_greedy_safe_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_n20_1 args.al_args.al_seed=1" # 4

    # "tobacco3482_resnet50_least_confidence_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=least_confidence_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=least_confidence_dropout_25_1 args.al_args.al_seed=1" # 1

    # "tobacco3482_resnet50_least_confidence_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=least_confidence args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=least_confidence_25_1 args.al_args.al_seed=1" # 1

    # "tobacco3482_resnet50_margin_sampling_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=margin_sampling_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=margin_sampling_dropout_25_1 args.al_args.al_seed=1" # 1

    # "tobacco3482_resnet50_margin_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=margin_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=margin_sampling_25_1 args.al_args.al_seed=1" # 1

    # "tobacco3482_resnet50_random_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=random_sampling_125_1 args.al_args.al_seed=1" # 0
    # "tobacco3482_resnet50_random_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=random_sampling_25_1 args.al_args.al_seed=1" # 1
    # "tobacco3482_resnet50_random_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=random_sampling_5_1 args.al_args.al_seed=1" # 2
    # "tobacco3482_resnet50_random_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_imb_2_1 args.al_args.al_seed=1" # 3
    # "tobacco3482_resnet50_random_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=random_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=random_sampling_25_imb_4_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_random_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_n10_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_random_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_n20_1 args.al_args.al_seed=1" # 4

    # "tobacco3482_resnet50_var_ratio_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=var_ratio args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=var_ratio_25_1 args.al_args.al_seed=1" # 1

    # "tobacco3482_resnet50_bald_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bald_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=bald_dropout_25_2 args.al_args.al_seed=2" # 1

    # "tobacco3482_resnet50_ceal_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=ceal_125_2 args.al_args.al_seed=2" # 0
    # "tobacco3482_resnet50_ceal_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=ceal_25_2 args.al_args.al_seed=2" # 1
    # "tobacco3482_resnet50_ceal_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=ceal_5_2 args.al_args.al_seed=2" # 2
    # "tobacco3482_resnet50_ceal_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=ceal args.training_args.experiment_name=ceal_25_imb_2_2 args.al_args.al_seed=2" # 3
    # "tobacco3482_resnet50_ceal_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=ceal args.al_args.n_classes_removed=4 args.training_args.experiment_name=ceal_25_imb_4_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_ceal_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=ceal args.training_args.experiment_name=ceal_25_n10_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_ceal_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=ceal args.training_args.experiment_name=ceal_25_n20_2 args.al_args.al_seed=2" # 4

    # "tobacco3482_resnet50_entropy_sampling_dropout_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=entropy_sampling_dropout_125_2 args.al_args.al_seed=2" # 0
    # "tobacco3482_resnet50_entropy_sampling_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=entropy_sampling_dropout_25_2 args.al_args.al_seed=2" # 1
    # "tobacco3482_resnet50_entropy_sampling_dropout_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=entropy_sampling_dropout_5_2 args.al_args.al_seed=2" # 2
    # "tobacco3482_resnet50_entropy_sampling_dropout_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_imb_2_2 args.al_args.al_seed=2" # 3
    # "tobacco3482_resnet50_entropy_sampling_dropout_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=entropy_sampling_dropout_25_imb_4_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_entropy_sampling_dropout_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_n10_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_entropy_sampling_dropout_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_n20_2 args.al_args.al_seed=2" # 4

    # "tobacco3482_resnet50_entropy_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=entropy_sampling_125_2 args.al_args.al_seed=2" # 0
    # "tobacco3482_resnet50_entropy_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=entropy_sampling_25_2 args.al_args.al_seed=2" # 1
    # "tobacco3482_resnet50_entropy_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=entropy_sampling_5_2 args.al_args.al_seed=2" # 2
    # "tobacco3482_resnet50_entropy_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_imb_2_2 args.al_args.al_seed=2" # 3
    # "tobacco3482_resnet50_entropy_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=entropy_sampling_25_imb_4_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_entropy_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_n10_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_entropy_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_n20_2 args.al_args.al_seed=2" # 4

    # "tobacco3482_resnet50_kcenter_greedy_safe_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=kcenter_greedy_safe_125_2 args.al_args.al_seed=2" # 0
    # "tobacco3482_resnet50_kcenter_greedy_safe_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=kcenter_greedy_safe_25_2 args.al_args.al_seed=2" # 1
    # "tobacco3482_resnet50_kcenter_greedy_safe_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=kcenter_greedy_safe_5_2 args.al_args.al_seed=2" # 2
    # "tobacco3482_resnet50_kcenter_greedy_safe_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_imb_2_2 args.al_args.al_seed=2" # 3
    # "tobacco3482_resnet50_kcenter_greedy_safe_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=kcenter_greedy_safe args.al_args.n_classes_removed=4 args.training_args.experiment_name=kcenter_greedy_safe_25_imb_4_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_kcenter_greedy_safe_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_n10_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_kcenter_greedy_safe_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_n20_2 args.al_args.al_seed=2" # 4

    # "tobacco3482_resnet50_least_confidence_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=least_confidence_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=least_confidence_dropout_25_2 args.al_args.al_seed=2" # 1

    # "tobacco3482_resnet50_least_confidence_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=least_confidence args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=least_confidence_25_2 args.al_args.al_seed=2" # 1

    # "tobacco3482_resnet50_margin_sampling_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=margin_sampling_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=margin_sampling_dropout_25_2 args.al_args.al_seed=2" # 1

    # "tobacco3482_resnet50_margin_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=margin_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=margin_sampling_25_2 args.al_args.al_seed=2" # 1

    # "tobacco3482_resnet50_random_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=random_sampling_125_2 args.al_args.al_seed=2" # 0
    # "tobacco3482_resnet50_random_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=random_sampling_25_2 args.al_args.al_seed=2" # 1
    # "tobacco3482_resnet50_random_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=random_sampling_5_2 args.al_args.al_seed=2" # 2
    # "tobacco3482_resnet50_random_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_imb_2_2 args.al_args.al_seed=2" # 3
    # "tobacco3482_resnet50_random_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=random_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=random_sampling_25_imb_4_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_random_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_n10_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_random_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_n20_2 args.al_args.al_seed=2" # 4

    # "tobacco3482_resnet50_var_ratio_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=var_ratio args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=var_ratio_25_2 args.al_args.al_seed=2" # 1


    # "tobacco3482_resnet50_bald_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bald_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=bald_dropout_25_3 args.al_args.al_seed=3" # 1

    # "tobacco3482_resnet50_ceal_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=ceal_125_3 args.al_args.al_seed=3" # 0
    # "tobacco3482_resnet50_ceal_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=ceal_25_3 args.al_args.al_seed=3" # 1
    # "tobacco3482_resnet50_ceal_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=ceal_5_3 args.al_args.al_seed=3" # 2
    # "tobacco3482_resnet50_ceal_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=ceal args.training_args.experiment_name=ceal_25_imb_2_3 args.al_args.al_seed=3" # 3
    # "tobacco3482_resnet50_ceal_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=ceal args.al_args.n_classes_removed=4 args.training_args.experiment_name=ceal_25_imb_4_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_ceal_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=ceal args.training_args.experiment_name=ceal_25_n10_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_ceal_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=ceal args.training_args.experiment_name=ceal_25_n20_3 args.al_args.al_seed=3" # 4

    # "tobacco3482_resnet50_entropy_sampling_dropout_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=entropy_sampling_dropout_125_3 args.al_args.al_seed=3" # 0
    # "tobacco3482_resnet50_entropy_sampling_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=entropy_sampling_dropout_25_3 args.al_args.al_seed=3" # 1
    # "tobacco3482_resnet50_entropy_sampling_dropout_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=entropy_sampling_dropout_5_3 args.al_args.al_seed=3" # 2
    # "tobacco3482_resnet50_entropy_sampling_dropout_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_imb_2_3 args.al_args.al_seed=3" # 3
    # "tobacco3482_resnet50_entropy_sampling_dropout_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=entropy_sampling_dropout_25_imb_4_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_entropy_sampling_dropout_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_n10_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_entropy_sampling_dropout_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_n20_3 args.al_args.al_seed=3" # 4

    # "tobacco3482_resnet50_entropy_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=entropy_sampling_125_3 args.al_args.al_seed=3" # 0
    # "tobacco3482_resnet50_entropy_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=entropy_sampling_25_3 args.al_args.al_seed=3" # 1
    # "tobacco3482_resnet50_entropy_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=entropy_sampling_5_3 args.al_args.al_seed=3" # 2
    # "tobacco3482_resnet50_entropy_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_imb_2_3 args.al_args.al_seed=3" # 3
    # "tobacco3482_resnet50_entropy_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=entropy_sampling_25_imb_4_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_entropy_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_n10_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_entropy_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_n20_3 args.al_args.al_seed=3" # 4

    # "tobacco3482_resnet50_kcenter_greedy_safe_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=kcenter_greedy_safe_125_3 args.al_args.al_seed=3" # 0
    # "tobacco3482_resnet50_kcenter_greedy_safe_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=kcenter_greedy_safe_25_3 args.al_args.al_seed=3" # 1
    # "tobacco3482_resnet50_kcenter_greedy_safe_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=kcenter_greedy_safe_5_3 args.al_args.al_seed=3" # 2
    # "tobacco3482_resnet50_kcenter_greedy_safe_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_imb_2_3 args.al_args.al_seed=3" # 3
    # "tobacco3482_resnet50_kcenter_greedy_safe_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=kcenter_greedy_safe args.al_args.n_classes_removed=4 args.training_args.experiment_name=kcenter_greedy_safe_25_imb_4_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_kcenter_greedy_safe_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_n10_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_kcenter_greedy_safe_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_n20_3 args.al_args.al_seed=3" # 4

    # "tobacco3482_resnet50_least_confidence_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=least_confidence_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=least_confidence_dropout_25_3 args.al_args.al_seed=3" # 1

    # "tobacco3482_resnet50_least_confidence_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=least_confidence args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=least_confidence_25_3 args.al_args.al_seed=3" # 1

    # "tobacco3482_resnet50_margin_sampling_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=margin_sampling_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=margin_sampling_dropout_25_3 args.al_args.al_seed=3" # 1

    # "tobacco3482_resnet50_margin_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=margin_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=margin_sampling_25_3 args.al_args.al_seed=3" # 1

    # "tobacco3482_resnet50_random_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=random_sampling_125_3 args.al_args.al_seed=3" # 0
    # "tobacco3482_resnet50_random_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=random_sampling_25_3 args.al_args.al_seed=3" # 1
    # "tobacco3482_resnet50_random_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=random_sampling_5_3 args.al_args.al_seed=3" # 2
    # "tobacco3482_resnet50_random_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_imb_2_3 args.al_args.al_seed=3" # 3
    # "tobacco3482_resnet50_random_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=random_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=random_sampling_25_imb_4_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_random_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_n10_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_random_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_n20_3 args.al_args.al_seed=3" # 4

    # "tobacco3482_resnet50_var_ratio_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=var_ratio args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=var_ratio_25_3 args.al_args.al_seed=3" # 1


    # "tobacco3482_resnet50_bald_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=bald_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=bald_dropout_25_4 args.al_args.al_seed=4" # 1

    # "tobacco3482_resnet50_ceal_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=ceal_125_4 args.al_args.al_seed=4" # 0
    # "tobacco3482_resnet50_ceal_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=ceal_25_4 args.al_args.al_seed=4" # 1
    # "tobacco3482_resnet50_ceal_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=ceal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=ceal_5_4 args.al_args.al_seed=4" # 2
    # "tobacco3482_resnet50_ceal_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=ceal args.training_args.experiment_name=ceal_25_imb_2_4 args.al_args.al_seed=4" # 3
    # "tobacco3482_resnet50_ceal_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=ceal args.al_args.n_classes_removed=4 args.training_args.experiment_name=ceal_25_imb_4_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_ceal_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=ceal args.training_args.experiment_name=ceal_25_n10_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_ceal_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=ceal args.training_args.experiment_name=ceal_25_n20_4 args.al_args.al_seed=4" # 4

    # "tobacco3482_resnet50_entropy_sampling_dropout_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=entropy_sampling_dropout_125_4 args.al_args.al_seed=4" # 0
    # "tobacco3482_resnet50_entropy_sampling_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=entropy_sampling_dropout_25_4 args.al_args.al_seed=4" # 1
    # "tobacco3482_resnet50_entropy_sampling_dropout_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling_dropout args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=entropy_sampling_dropout_5_4 args.al_args.al_seed=4" # 2
    # "tobacco3482_resnet50_entropy_sampling_dropout_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_imb_2_4 args.al_args.al_seed=4" # 3
    # "tobacco3482_resnet50_entropy_sampling_dropout_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=entropy_sampling_dropout_25_imb_4_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_entropy_sampling_dropout_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_n10_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_entropy_sampling_dropout_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=entropy_sampling_dropout args.training_args.experiment_name=entropy_sampling_dropout_25_n20_4 args.al_args.al_seed=4" # 4

    # "tobacco3482_resnet50_entropy_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=entropy_sampling_125_4 args.al_args.al_seed=4" # 0
    # "tobacco3482_resnet50_entropy_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=entropy_sampling_25_4 args.al_args.al_seed=4" # 1
    # "tobacco3482_resnet50_entropy_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=entropy_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=entropy_sampling_5_4 args.al_args.al_seed=4" # 2
    # "tobacco3482_resnet50_entropy_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_imb_2_4 args.al_args.al_seed=4" # 3
    # "tobacco3482_resnet50_entropy_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=entropy_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=entropy_sampling_25_imb_4_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_entropy_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_n10_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_entropy_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=entropy_sampling args.training_args.experiment_name=entropy_sampling_25_n20_4 args.al_args.al_seed=4" # 4

    # "tobacco3482_resnet50_kcenter_greedy_safe_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=kcenter_greedy_safe_125_4 args.al_args.al_seed=4" # 0
    # "tobacco3482_resnet50_kcenter_greedy_safe_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=kcenter_greedy_safe_25_4 args.al_args.al_seed=4" # 1
    # "tobacco3482_resnet50_kcenter_greedy_safe_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=kcenter_greedy_safe args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=kcenter_greedy_safe_5_4 args.al_args.al_seed=4" # 2
    # "tobacco3482_resnet50_kcenter_greedy_safe_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_imb_2_4 args.al_args.al_seed=4" # 3
    # "tobacco3482_resnet50_kcenter_greedy_safe_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=kcenter_greedy_safe args.al_args.n_classes_removed=4 args.training_args.experiment_name=kcenter_greedy_safe_25_imb_4_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_kcenter_greedy_safe_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_n10_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_kcenter_greedy_safe_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=kcenter_greedy_safe args.training_args.experiment_name=kcenter_greedy_safe_25_n20_4 args.al_args.al_seed=4" # 4

    # "tobacco3482_resnet50_least_confidence_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=least_confidence_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=least_confidence_dropout_25_4 args.al_args.al_seed=4" # 1

    # "tobacco3482_resnet50_least_confidence_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=least_confidence args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=least_confidence_25_4 args.al_args.al_seed=4" # 1

    # "tobacco3482_resnet50_margin_sampling_dropout_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=margin_sampling_dropout args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=margin_sampling_dropout_25_4 args.al_args.al_seed=4" # 1

    # "tobacco3482_resnet50_margin_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=margin_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=margin_sampling_25_4 args.al_args.al_seed=4" # 1

    # "tobacco3482_resnet50_random_sampling_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=random_sampling_125_4 args.al_args.al_seed=4" # 0
    # "tobacco3482_resnet50_random_sampling_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=random_sampling_25_4 args.al_args.al_seed=4" # 1
    # "tobacco3482_resnet50_random_sampling_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=random_sampling args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=random_sampling_5_4 args.al_args.al_seed=4" # 2
    # "tobacco3482_resnet50_random_sampling_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_imb_2_4 args.al_args.al_seed=4" # 3
    # "tobacco3482_resnet50_random_sampling_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=random_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=random_sampling_25_imb_4_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_random_sampling_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_n10_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_random_sampling_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=random_sampling args.training_args.experiment_name=random_sampling_25_n20_4 args.al_args.al_seed=4" # 4

    # "tobacco3482_resnet50_var_ratio_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=var_ratio args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=var_ratio_25_4 args.al_args.al_seed=4" # 1


    # "tobacco3482_resnet50_adv_bim_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=adv_bim_125_1 args.al_args.al_seed=1" # 0
    # "tobacco3482_resnet50_adv_bim_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=adv_bim_25_1 args.al_args.al_seed=1" # 1
    # "tobacco3482_resnet50_adv_bim_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=adv_bim_5_1 args.al_args.al_seed=1" # 2
    # "tobacco3482_resnet50_adv_bim_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_imb_2_1 args.al_args.al_seed=1" # 3
    # "tobacco3482_resnet50_adv_bim_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=adv_bim_48 args.al_args.n_classes_removed=4 args.training_args.experiment_name=adv_bim_25_imb_4_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_adv_bim_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_n10_1 args.al_args.al_seed=1" # 4
    # "tobacco3482_resnet50_adv_bim_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_n20_1 args.al_args.al_seed=1" # 4

    # "tobacco3482_resnet50_adv_bim_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=adv_bim_125_2 args.al_args.al_seed=2" # 0
    # "tobacco3482_resnet50_adv_bim_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=adv_bim_25_2 args.al_args.al_seed=2" # 1
    # "tobacco3482_resnet50_adv_bim_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=adv_bim_5_2 args.al_args.al_seed=2" # 2
    # "tobacco3482_resnet50_adv_bim_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_imb_2_2 args.al_args.al_seed=2" # 3
    # "tobacco3482_resnet50_adv_bim_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=adv_bim_48 args.al_args.n_classes_removed=4 args.training_args.experiment_name=adv_bim_25_imb_4_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_adv_bim_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_n10_2 args.al_args.al_seed=2" # 4
    # "tobacco3482_resnet50_adv_bim_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_n20_2 args.al_args.al_seed=2" # 4

    # "tobacco3482_resnet50_adv_bim_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=adv_bim_125_3 args.al_args.al_seed=3" # 0
    # "tobacco3482_resnet50_adv_bim_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=adv_bim_25_3 args.al_args.al_seed=3" # 1
    # "tobacco3482_resnet50_adv_bim_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=adv_bim_5_3 args.al_args.al_seed=3" # 2
    # "tobacco3482_resnet50_adv_bim_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_imb_2_3 args.al_args.al_seed=3" # 3
    # "tobacco3482_resnet50_adv_bim_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=adv_bim_48 args.al_args.n_classes_removed=4 args.training_args.experiment_name=adv_bim_25_imb_4_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_adv_bim_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_n10_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_adv_bim_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_n20_3 args.al_args.al_seed=3" # 4
    # "tobacco3482_resnet50_adv_bim_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=adv_bim_125_4 args.al_args.al_seed=4" # 0
    # "tobacco3482_resnet50_adv_bim_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=adv_bim_25_4 args.al_args.al_seed=4" # 1
    # "tobacco3482_resnet50_adv_bim_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=adv_bim_48 args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=adv_bim_5_4 args.al_args.al_seed=4" # 2
    # "tobacco3482_resnet50_adv_bim_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_imb_2_4 args.al_args.al_seed=4" # 3
    # "tobacco3482_resnet50_adv_bim_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=adv_bim_48 args.al_args.n_classes_removed=4 args.training_args.experiment_name=adv_bim_25_imb_4_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_adv_bim_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_n10_4 args.al_args.al_seed=4" # 4
    # "tobacco3482_resnet50_adv_bim_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=adv_bim_48 args.training_args.experiment_name=adv_bim_25_n20_4 args.al_args.al_seed=4" # 4


    # "tobacco3482_resnet50_vaal_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=vaal_125 args.al_args.al_seed=0" # 0
    # "tobacco3482_resnet50_vaal_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=vaal_25 args.al_args.al_seed=0" # 1
    # "tobacco3482_resnet50_vaal_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=vaal_5 args.al_args.al_seed=0" # 2
    # "tobacco3482_resnet50_vaal_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=vaal args.training_args.experiment_name=vaal_25_imb_2 args.al_args.al_seed=0" # 3
    # "tobacco3482_resnet50_vaal_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=vaal args.al_args.n_classes_removed=4 args.training_args.experiment_name=vaal_25_imb_4 args.al_args.al_seed=0" # 4
    # "tobacco3482_resnet50_vaal_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=vaal args.training_args.experiment_name=vaal_25_n10 args.al_args.al_seed=0" # 4
    # "tobacco3482_resnet50_vaal_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=vaal args.training_args.experiment_name=vaal_25_n20 args.al_args.al_seed=0" #

    "tobacco3482_resnet50_vaal_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=vaal_125_1 args.al_args.al_seed=1" # 0
    "tobacco3482_resnet50_vaal_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=vaal_25_1 args.al_args.al_seed=1" # 1
    "tobacco3482_resnet50_vaal_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=vaal_5_1 args.al_args.al_seed=1" # 2
    "tobacco3482_resnet50_vaal_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=vaal args.training_args.experiment_name=vaal_25_imb_2_1 args.al_args.al_seed=1" # 3
    "tobacco3482_resnet50_vaal_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=vaal args.al_args.n_classes_removed=4 args.training_args.experiment_name=vaal_25_imb_4_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_vaal_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=vaal args.training_args.experiment_name=vaal_25_n10_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_vaal_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=vaal args.training_args.experiment_name=vaal_25_n20_1 args.al_args.al_seed=1" #

    "tobacco3482_resnet50_vaal_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=vaal_125_2 args.al_args.al_seed=2" # 0
    "tobacco3482_resnet50_vaal_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=vaal_25_2 args.al_args.al_seed=2" # 1
    "tobacco3482_resnet50_vaal_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=vaal_5_2 args.al_args.al_seed=2" # 2
    "tobacco3482_resnet50_vaal_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=vaal args.training_args.experiment_name=vaal_25_imb_2_2 args.al_args.al_seed=2" # 3
    "tobacco3482_resnet50_vaal_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=vaal args.al_args.n_classes_removed=4 args.training_args.experiment_name=vaal_25_imb_4_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_vaal_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=vaal args.training_args.experiment_name=vaal_25_n10_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_vaal_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=vaal args.training_args.experiment_name=vaal_25_n20_2 args.al_args.al_seed=2" #

    "tobacco3482_resnet50_vaal_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=vaal_125_3 args.al_args.al_seed=3" # 0
    "tobacco3482_resnet50_vaal_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=vaal_25_3 args.al_args.al_seed=3" # 1
    "tobacco3482_resnet50_vaal_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=vaal_5_3 args.al_args.al_seed=3" # 2
    "tobacco3482_resnet50_vaal_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=vaal args.training_args.experiment_name=vaal_25_imb_2_3 args.al_args.al_seed=3" # 3
    "tobacco3482_resnet50_vaal_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=vaal args.al_args.n_classes_removed=4 args.training_args.experiment_name=vaal_25_imb_4_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_vaal_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=vaal args.training_args.experiment_name=vaal_25_n10_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_vaal_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=vaal args.training_args.experiment_name=vaal_25_n20_3 args.al_args.al_seed=3" #

    "tobacco3482_resnet50_vaal_125 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name=vaal_125_4 args.al_args.al_seed=4" # 0
    "tobacco3482_resnet50_vaal_25 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name=vaal_25_4 args.al_args.al_seed=4" # 1
    "tobacco3482_resnet50_vaal_5 $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=vaal args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name=vaal_5_4 args.al_args.al_seed=4" # 2
    "tobacco3482_resnet50_vaal_imb_2 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=vaal args.training_args.experiment_name=vaal_25_imb_2_4 args.al_args.al_seed=4" # 3
    "tobacco3482_resnet50_vaal_imb_4 $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=vaal args.al_args.n_classes_removed=4 args.training_args.experiment_name=vaal_25_imb_4_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_vaal_n10 $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=vaal args.training_args.experiment_name=vaal_25_n10_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_vaal_n20 $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=vaal args.training_args.experiment_name=vaal_25_n20_4 args.al_args.al_seed=4" # 4
)