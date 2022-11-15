#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_EXP_CONFIG_TOBACCO3482="$SCRIPT_DIR/../../al_train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/tobacco3482/resnet50"
BASE_EXP_CONFIG_TOBACCO3482_IMB="$SCRIPT_DIR/../../al_train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/tobacco3482/resnet50_imb"
BASE_EXP_CONFIG_TOBACCO3482_N10="$SCRIPT_DIR/../../al_train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/tobacco3482/resnet50_noisy_10"
BASE_EXP_CONFIG_TOBACCO3482_N20="$SCRIPT_DIR/../../al_train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/tobacco3482/resnet50_noisy_20"

usage()
{
    echo "Usage:"
    echo "./exps.sh <query_strategy> <experiment> <path-to-data>"
}

# set query strategy
QUERY_STRATEGY=$1
EXPERIMENT=$2
PATH_TO_DATASET=$3

if ! [[ "$EXPERIMENT" =~ ^(125|25|5|imb_2|imb_4|n10|n20)$ ]]; then
  exit 1
fi

if [ "$QUERY_STRATEGY" = "" ]; then
  usage
  exit 1
fi

if [[ "$QUERY_STRATEGY" == "lpl" ]]; then 
    # experiment config for query size 1.25%
    $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name="${QUERY_STRATEGY}_125" args.al_args.al_seed=0

    # experiment config for query size 2.5%
    $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name="${QUERY_STRATEGY}_25" args.al_args.al_seed=0

    # experiment config for query size 5.0%
    $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name="${QUERY_STRATEGY}_5" args.al_args.al_seed=0

    # experiment config for query size 2.5% and data imbalance m=2
    $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_2" args.al_args.al_seed=0

    # experiment config for query size 2.5% and data imbalance m=4
    $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=$QUERY_STRATEGY args.al_args.n_classes_removed=4 args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_4" args.al_args.al_seed=0

    # experiment config for query size 2.5% and noise = 10%
    $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n10" args.al_args.al_seed=0

    # experiment config for query size 2.5% and noise = 20%
    $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n20" args.al_args.al_seed=0
elif [[ "$QUERY_STRATEGY" == "waal" ]]; then 
    # experiment config for query size 1.25%
    $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name="${QUERY_STRATEGY}_125" args.al_args.al_seed=0

    # experiment config for query size 2.5%
    $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name="${QUERY_STRATEGY}_25" args.al_args.al_seed=0

    # experiment config for query size 5.0%
    $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name="${QUERY_STRATEGY}_5" args.al_args.al_seed=0

    # experiment config for query size 2.5% and data imbalance m=2
    $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_2" args.al_args.al_seed=0

    # experiment config for query size 2.5% and data imbalance m=4
    $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=$QUERY_STRATEGY args.al_args.n_classes_removed=4 args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_4" args.al_args.al_seed=0

    # experiment config for query size 2.5% and noise = 10%
    $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n10" args.al_args.al_seed=0

    # experiment config for query size 2.5% and noise = 20%
    $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n20" args.al_args.al_seed=0
else
    if [[ "$EXPERIMENT" == "125" ]]; then 
        # experiment config for query size 1.25%
        $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name="${QUERY_STRATEGY}_125" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi

    if [[ "$EXPERIMENT" == "25" ]]; then 
        # experiment config for query size 2.5%
        $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name="${QUERY_STRATEGY}_25" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi 

    if [[ "$EXPERIMENT" == "5" ]]; then 
        # experiment config for query size 5.0%
        $BASE_EXP_CONFIG_TOBACCO3482 args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name="${QUERY_STRATEGY}_5" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi

    if [[ "$EXPERIMENT" == "imb_2" ]]; then 
        # experiment config for query size 2.5% and data imbalance m=2
        $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_2" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi 

    if [[ "$EXPERIMENT" == "imb_4" ]]; then 
        # experiment config for query size 2.5% and data imbalance m=4
        $BASE_EXP_CONFIG_TOBACCO3482_IMB args/al_args=$QUERY_STRATEGY args.al_args.n_classes_removed=4 args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_4" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi 

    if [[ "$EXPERIMENT" == "n10" ]]; then 
        # experiment config for query size 2.5% and noise = 10%
        $BASE_EXP_CONFIG_TOBACCO3482_N10 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n10" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi 

    if [[ "$EXPERIMENT" == "n20" ]]; then 
        # experiment config for query size 2.5% and noise = 20%
        $BASE_EXP_CONFIG_TOBACCO3482_N20 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n20" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi
fi