#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TRAIN_SCRIPT="$SCRIPT_DIR/../al_train.sh"
BASE_EXP_CONFIG="$TRAIN_SCRIPT --config-path ../../../cfg/al_adv/ +experiment=active_learning/tobacco3482_pre/resnet50"
BASE_EXP_CONFIG_IMB="$TRAIN_SCRIPT --config-path ../../../cfg/al_adv/ +experiment=active_learning/tobacco3482_pre/resnet50_imb"
BASE_EXP_CONFIG_N10="$TRAIN_SCRIPT --config-path ../../../cfg/al_adv/ +experiment=active_learning/tobacco3482_pre/resnet50_noisy_10"
BASE_EXP_CONFIG_N20="$TRAIN_SCRIPT --config-path ../../../cfg/al_adv/ +experiment=active_learning/tobacco3482_pre/resnet50_noisy_20"

# set default parameters
QUERY_STRATEGY=random_sampling
EXPERIMENT=25
PATH_TO_DATASET=
SEED=0

usage()
{
    echo "Usage:"
    echo "./sbatch_run.sh -q=<query-strategy> --exp=<experiment-type> --data-path=</path/to/dataset> --seed=<seed>"
    echo ""
    echo " -q : Query strategy"
    echo " --exp : Experiment type. Choices: [125|25|5|imb_2|imb_4|n10|n20]"
    echo " --data-path : Path to dataset directory."
    echo " --seed : Random seed."
    echo " -h | --help : Displays the help"
    echo ""
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --q)
            QUERY_STRATEGY=$VALUE
            ;;
        --exp)
            EXPERIMENT=$VALUE
            ;;
        --data-path)
            PATH_TO_DATASET=$VALUE
            ;;
        --seed)
            SEED=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

if ! [[ "$EXPERIMENT" =~ ^(125|25|5|imb_2|imb_4|n10|n20)$ ]]; then
  exit 1
fi

if [ "$QUERY_STRATEGY" = "" ]; then
  usage
  exit 1
fi

if [[ "$QUERY_STRATEGY" == "lpl" ]]; then 
    if [[ "$EXPERIMENT" == "125" ]]; then 
        # experiment config for query size 1.25%
        $BASE_EXP_CONFIG args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name="${QUERY_STRATEGY}_125" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True
    fi

    if [[ "$EXPERIMENT" == "25" ]]; then 
        # experiment config for query size 2.5%
        $BASE_EXP_CONFIG args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name="${QUERY_STRATEGY}_25" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True
    fi 

    if [[ "$EXPERIMENT" == "5" ]]; then 
        # experiment config for query size 5.0%
        $BASE_EXP_CONFIG args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name="${QUERY_STRATEGY}_5" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True
    fi

    if [[ "$EXPERIMENT" == "imb_2" ]]; then 
        # experiment config for query size 2.5% and data imbalance m=2
        $BASE_EXP_CONFIG_IMB args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_2" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True
    fi 

    if [[ "$EXPERIMENT" == "imb_4" ]]; then 
        # experiment config for query size 2.5% and data imbalance m=4
        $BASE_EXP_CONFIG_IMB args/al_args=$QUERY_STRATEGY args.al_args.n_classes_removed=4 args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_4" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True
    fi 

    if [[ "$EXPERIMENT" == "n10" ]]; then 
        # experiment config for query size 2.5% and noise = 10%
        $BASE_EXP_CONFIG_N10 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n10" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True
    fi 

    if [[ "$EXPERIMENT" == "n20" ]]; then 
        # experiment config for query size 2.5% and noise = 20%
        $BASE_EXP_CONFIG_N20 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n20" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True
    fi
elif [[ "$QUERY_STRATEGY" == "waal" ]]; then 
    if [[ "$EXPERIMENT" == "125" ]]; then 
        # experiment config for query size 1.25%
        $BASE_EXP_CONFIG args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name="${QUERY_STRATEGY}_125" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=waal args.model_args.name=resnet50_waal 
    fi

    if [[ "$EXPERIMENT" == "25" ]]; then 
        # experiment config for query size 2.5%
        $BASE_EXP_CONFIG args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name="${QUERY_STRATEGY}_25" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=waal args.model_args.name=resnet50_waal 
    fi 

    if [[ "$EXPERIMENT" == "5" ]]; then 
        # experiment config for query size 5.0%
        $BASE_EXP_CONFIG args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name="${QUERY_STRATEGY}_5" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=waal args.model_args.name=resnet50_waal 
    fi

    if [[ "$EXPERIMENT" == "imb_2" ]]; then 
        # experiment config for query size 2.5% and data imbalance m=2
        $BASE_EXP_CONFIG_IMB args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_2" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=waal args.model_args.name=resnet50_waal 
    fi 

    if [[ "$EXPERIMENT" == "imb_4" ]]; then 
        # experiment config for query size 2.5% and data imbalance m=4
        $BASE_EXP_CONFIG_IMB args/al_args=$QUERY_STRATEGY args.al_args.n_classes_removed=4 args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_4" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=waal args.model_args.name=resnet50_waal 
    fi 

    if [[ "$EXPERIMENT" == "n10" ]]; then 
        # experiment config for query size 2.5% and noise = 10%
        $BASE_EXP_CONFIG_N10 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n10" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=waal args.model_args.name=resnet50_waal 
    fi 

    if [[ "$EXPERIMENT" == "n20" ]]; then 
        # experiment config for query size 2.5% and noise = 20%
        $BASE_EXP_CONFIG_N20 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n20" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET args/training_args=waal args.model_args.name=resnet50_waal 
    fi
else
    if [[ "$EXPERIMENT" == "125" ]]; then 
        # experiment config for query size 1.25%
        $BASE_EXP_CONFIG args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=29 args.training_args.experiment_name="${QUERY_STRATEGY}_125" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi

    if [[ "$EXPERIMENT" == "25" ]]; then 
        # experiment config for query size 2.5%
        $BASE_EXP_CONFIG args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=15 args.training_args.experiment_name="${QUERY_STRATEGY}_25" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi 

    if [[ "$EXPERIMENT" == "5" ]]; then 
        # experiment config for query size 5.0%
        $BASE_EXP_CONFIG args/al_args=$QUERY_STRATEGY args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=8 args.training_args.experiment_name="${QUERY_STRATEGY}_5" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi

    if [[ "$EXPERIMENT" == "imb_2" ]]; then 
        # experiment config for query size 2.5% and data imbalance m=2
        $BASE_EXP_CONFIG_IMB args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_2" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi 

    if [[ "$EXPERIMENT" == "imb_4" ]]; then 
        # experiment config for query size 2.5% and data imbalance m=4
        $BASE_EXP_CONFIG_IMB args/al_args=$QUERY_STRATEGY args.al_args.n_classes_removed=4 args.training_args.experiment_name="${QUERY_STRATEGY}_25_imb_4" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi 

    if [[ "$EXPERIMENT" == "n10" ]]; then 
        # experiment config for query size 2.5% and noise = 10%
        $BASE_EXP_CONFIG_N10 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n10" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi 

    if [[ "$EXPERIMENT" == "n20" ]]; then 
        # experiment config for query size 2.5% and noise = 20%
        $BASE_EXP_CONFIG_N20 args/al_args=$QUERY_STRATEGY args.training_args.experiment_name="${QUERY_STRATEGY}_25_n20" args.al_args.al_seed=0 args.data_args.dataset_dir=$PATH_TO_DATASET
    fi
fi