#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_EXP_CONFIG_rvlcdip="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50"
BASE_EXP_CONFIG_rvlcdip_IMB="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50_imb"
BASE_EXP_CONFIG_rvlcdip_N10="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50_noisy_10"
BASE_EXP_CONFIG_rvlcdip_N20="$SCRIPT_DIR/../../train.sh --config-path ../../../cfg/al_adv/ +experiment=active_learning/new_rvlcdip/resnet50_noisy_20"
CONFIGS=(
    "rvlcdip_resnet50_lpl_125 $BASE_EXP_CONFIG_rvlcdip args/al_args=lpl args.al_args.n_query_ratio=0.0125 args.al_args.n_rounds=25 args.training_args.experiment_name=lpl_125_0 args.al_args.al_seed=0 args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True" # 0
    "rvlcdip_resnet50_lpl_25 $BASE_EXP_CONFIG_rvlcdip args/al_args=lpl args.al_args.n_query_ratio=0.025 args.al_args.n_rounds=13 args.training_args.experiment_name=lpl_25_0 args.al_args.al_seed=0 args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True" # 1
    "rvlcdip_resnet50_lpl_5 $BASE_EXP_CONFIG_rvlcdip args/al_args=lpl args.al_args.n_query_ratio=0.05 args.al_args.n_rounds=7 args.training_args.experiment_name=lpl_5_0 args.al_args.al_seed=0 args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True" # 2
    "rvlcdip_resnet50_lpl_imb_2 $BASE_EXP_CONFIG_rvlcdip_IMB args/al_args=lpl args.training_args.experiment_name=lpl_25_imb_2_0 args.al_args.al_seed=0 args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True" # 3
    "rvlcdip_resnet50_lpl_imb_4 $BASE_EXP_CONFIG_rvlcdip_IMB args/al_args=lpl args.al_args.n_classes_removed=4 args.training_args.experiment_name=lpl_25_imb_4_0 args.al_args.al_seed=0 args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True" # 4
    "rvlcdip_resnet50_lpl_n10 $BASE_EXP_CONFIG_rvlcdip_N10 args/al_args=lpl args.training_args.experiment_name=lpl_25_n10_0 args.al_args.al_seed=0 args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True" # 4
    "rvlcdip_resnet50_lpl_n20 $BASE_EXP_CONFIG_rvlcdip_N20 args/al_args=lpl args.training_args.experiment_name=lpl_25_n20_0 args.al_args.al_seed=0 args/training_args=lpl args.model_args.name=resnet50_lpl args.data_args.data_loader_args.dataloader_drop_last=True" #
)