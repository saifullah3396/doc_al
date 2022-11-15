#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_EXP_CONFIG_tobacco3482="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_tobacco_pre/resnet50"
BASE_EXP_CONFIG_tobacco3482_IMB="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_tobacco_pre/resnet50_imb"
BASE_EXP_CONFIG_tobacco3482_N10="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_tobacco_pre/resnet50_noisy_10"
BASE_EXP_CONFIG_tobacco3482_N20="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_tobacco_pre/resnet50_noisy_20"
CONFIGS=(
    # start
    "tobacco3482_resnet50_margin_sampling_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_imb_2 args.al_args.al_seed=0" # 3
    "tobacco3482_resnet50_margin_sampling_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=margin_sampling_8k_imb_4 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_margin_sampling_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_n10 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_margin_sampling_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_n20 args.al_args.al_seed=0" # 4

    "tobacco3482_resnet50_margin_sampling_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_imb_2 args.al_args.al_seed=0" # 3
    "tobacco3482_resnet50_margin_sampling_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=margin_sampling_dropout_8k_imb_4 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_margin_sampling_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_n10 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_margin_sampling_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_n20 args.al_args.al_seed=0" # 4

    "tobacco3482_resnet50_least_confidence_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_imb_2 args.al_args.al_seed=0" # 3
    "tobacco3482_resnet50_least_confidence_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence args.al_args.n_classes_removed=4 args.training_args.experiment_name=least_confidence_8k_imb_4 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_least_confidence_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_n10 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_least_confidence_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_n20 args.al_args.al_seed=0" # 4

    "tobacco3482_resnet50_least_confidence_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_imb_2 args.al_args.al_seed=0" # 3
    "tobacco3482_resnet50_least_confidence_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=least_confidence_dropout_8k_imb_4 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_least_confidence_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_n10 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_least_confidence_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_n20 args.al_args.al_seed=0" # 4

    "tobacco3482_resnet50_bald_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_imb_2 args.al_args.al_seed=0" # 3
    "tobacco3482_resnet50_bald_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=bald_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=bald_dropout_8k_imb_4 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_bald_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_n10 args.al_args.al_seed=0" # 4
    "tobacco3482_resnet50_bald_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_n20 args.al_args.al_seed=0" # 4


    # start
    "tobacco3482_resnet50_margin_sampling_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_imb_2_1 args.al_args.al_seed=1" # 3
    "tobacco3482_resnet50_margin_sampling_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=margin_sampling_8k_imb_4_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_margin_sampling_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_n10_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_margin_sampling_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_n20_1 args.al_args.al_seed=1" # 4

    "tobacco3482_resnet50_margin_sampling_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_imb_2_1 args.al_args.al_seed=1" # 3
    "tobacco3482_resnet50_margin_sampling_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=margin_sampling_dropout_8k_imb_4_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_margin_sampling_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_n10_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_margin_sampling_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_n20_1 args.al_args.al_seed=1" # 4

    "tobacco3482_resnet50_least_confidence_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_imb_2_1 args.al_args.al_seed=1" # 3
    "tobacco3482_resnet50_least_confidence_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence args.al_args.n_classes_removed=4 args.training_args.experiment_name=least_confidence_8k_imb_4_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_least_confidence_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_n10_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_least_confidence_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_n20_1 args.al_args.al_seed=1" # 4

    "tobacco3482_resnet50_least_confidence_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_imb_2_1 args.al_args.al_seed=1" # 3
    "tobacco3482_resnet50_least_confidence_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=least_confidence_dropout_8k_imb_4_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_least_confidence_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_n10_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_least_confidence_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_n20_1 args.al_args.al_seed=1" # 4

    "tobacco3482_resnet50_bald_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_imb_2_1 args.al_args.al_seed=1" # 3
    "tobacco3482_resnet50_bald_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=bald_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=bald_dropout_8k_imb_4_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_bald_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_n10_1 args.al_args.al_seed=1" # 4
    "tobacco3482_resnet50_bald_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_n20_1 args.al_args.al_seed=1" # 4


    # start
    "tobacco3482_resnet50_margin_sampling_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_imb_2_2 args.al_args.al_seed=2" # 3
    "tobacco3482_resnet50_margin_sampling_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=margin_sampling_8k_imb_4_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_margin_sampling_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_n10_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_margin_sampling_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_n20_2 args.al_args.al_seed=2" # 4

    "tobacco3482_resnet50_margin_sampling_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_imb_2_2 args.al_args.al_seed=2" # 3
    "tobacco3482_resnet50_margin_sampling_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=margin_sampling_dropout_8k_imb_4_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_margin_sampling_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_n10_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_margin_sampling_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_n20_2 args.al_args.al_seed=2" # 4

    "tobacco3482_resnet50_least_confidence_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_imb_2_2 args.al_args.al_seed=2" # 3
    "tobacco3482_resnet50_least_confidence_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence args.al_args.n_classes_removed=4 args.training_args.experiment_name=least_confidence_8k_imb_4_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_least_confidence_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_n10_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_least_confidence_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_n20_2 args.al_args.al_seed=2" # 4

    "tobacco3482_resnet50_least_confidence_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_imb_2_2 args.al_args.al_seed=2" # 3
    "tobacco3482_resnet50_least_confidence_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=least_confidence_dropout_8k_imb_4_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_least_confidence_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_n10_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_least_confidence_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_n20_2 args.al_args.al_seed=2" # 4

    "tobacco3482_resnet50_bald_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_imb_2_2 args.al_args.al_seed=2" # 3
    "tobacco3482_resnet50_bald_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=bald_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=bald_dropout_8k_imb_4_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_bald_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_n10_2 args.al_args.al_seed=2" # 4
    "tobacco3482_resnet50_bald_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_n20_2 args.al_args.al_seed=2" # 4

    # start
    "tobacco3482_resnet50_margin_sampling_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_imb_2_3 args.al_args.al_seed=3" # 3
    "tobacco3482_resnet50_margin_sampling_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=margin_sampling_8k_imb_4_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_margin_sampling_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_n10_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_margin_sampling_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_n20_3 args.al_args.al_seed=3" # 4

    "tobacco3482_resnet50_margin_sampling_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_imb_2_3 args.al_args.al_seed=3" # 3
    "tobacco3482_resnet50_margin_sampling_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=margin_sampling_dropout_8k_imb_4_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_margin_sampling_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_n10_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_margin_sampling_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_n20_3 args.al_args.al_seed=3" # 4

    "tobacco3482_resnet50_least_confidence_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_imb_2_3 args.al_args.al_seed=3" # 3
    "tobacco3482_resnet50_least_confidence_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence args.al_args.n_classes_removed=4 args.training_args.experiment_name=least_confidence_8k_imb_4_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_least_confidence_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_n10_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_least_confidence_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_n20_3 args.al_args.al_seed=3" # 4

    "tobacco3482_resnet50_least_confidence_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_imb_2_3 args.al_args.al_seed=3" # 3
    "tobacco3482_resnet50_least_confidence_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=least_confidence_dropout_8k_imb_4_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_least_confidence_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_n10_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_least_confidence_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_n20_3 args.al_args.al_seed=3" # 4

    "tobacco3482_resnet50_bald_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_imb_2_3 args.al_args.al_seed=3" # 3
    "tobacco3482_resnet50_bald_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=bald_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=bald_dropout_8k_imb_4_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_bald_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_n10_3 args.al_args.al_seed=3" # 4
    "tobacco3482_resnet50_bald_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_n20_3 args.al_args.al_seed=3" # 4

    # start
    "tobacco3482_resnet50_margin_sampling_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_imb_2_4 args.al_args.al_seed=4" # 3
    "tobacco3482_resnet50_margin_sampling_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling args.al_args.n_classes_removed=4 args.training_args.experiment_name=margin_sampling_8k_imb_4_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_margin_sampling_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_n10_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_margin_sampling_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=margin_sampling args.training_args.experiment_name=margin_sampling_8k_n20_4 args.al_args.al_seed=4" # 4

    "tobacco3482_resnet50_margin_sampling_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_imb_2_4 args.al_args.al_seed=4" # 3
    "tobacco3482_resnet50_margin_sampling_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=margin_sampling_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=margin_sampling_dropout_8k_imb_4_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_margin_sampling_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_n10_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_margin_sampling_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=margin_sampling_dropout args.training_args.experiment_name=margin_sampling_dropout_8k_n20_4 args.al_args.al_seed=4" # 4

    "tobacco3482_resnet50_least_confidence_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_imb_2_4 args.al_args.al_seed=4" # 3
    "tobacco3482_resnet50_least_confidence_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence args.al_args.n_classes_removed=4 args.training_args.experiment_name=least_confidence_8k_imb_4_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_least_confidence_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_n10_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_least_confidence_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=least_confidence args.training_args.experiment_name=least_confidence_8k_n20_4 args.al_args.al_seed=4" # 4

    "tobacco3482_resnet50_least_confidence_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_imb_2_4 args.al_args.al_seed=4" # 3
    "tobacco3482_resnet50_least_confidence_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=least_confidence_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=least_confidence_dropout_8k_imb_4_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_least_confidence_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_n10_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_least_confidence_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=least_confidence_dropout args.training_args.experiment_name=least_confidence_dropout_8k_n20_4 args.al_args.al_seed=4" # 4

    "tobacco3482_resnet50_bald_dropout_imb_2 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_imb_2_4 args.al_args.al_seed=4" # 3
    "tobacco3482_resnet50_bald_dropout_imb_4 $BASE_EXP_CONFIG_tobacco3482_IMB args/al_args=bald_dropout args.al_args.n_classes_removed=4 args.training_args.experiment_name=bald_dropout_8k_imb_4_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_bald_dropout_n10 $BASE_EXP_CONFIG_tobacco3482_N10 args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_n10_4 args.al_args.al_seed=4" # 4
    "tobacco3482_resnet50_bald_dropout_n20 $BASE_EXP_CONFIG_tobacco3482_N20 args/al_args=bald_dropout args.training_args.experiment_name=bald_dropout_8k_n20_4 args.al_args.al_seed=4" # 4
)