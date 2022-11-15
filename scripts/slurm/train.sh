#!/bin/bash
set -e -o pipefail

CONFIG_SH=$1
source "$PWD/$CONFIG_SH"
echo "Current working dir: $PWD"

# matches the slurm id and the setting number
echo "Task ID: $SLURM_ARRAY_TASK_ID SubTask: $SLURM_LOCALID"
SID=$SLURM_ARRAY_TASK_ID
SID=$(( SID+=$SLURM_LOCALID ))
mkdir -p "$XAI_TORCH_OUTPUT_DIR/slurm_logs"

# ADJUST define your setup logic
for CID in ${!CONFIGS[@]}
do
    # prints the settings if selected
    if [[ $SID -eq -1 || $CID -eq $SID ]]
    then
        set -- ${CONFIGS[CID]}
        EXP_NAME=$1
        SCRIPT="${@:2}"

        echo "=================================================="
        echo "CID: $CID"
        echo "Experiment: $EXP_NAME"
        echo "Task: $SCRIPT"
        echo "=================================================="
        $SCRIPT &> $XAI_TORCH_OUTPUT_DIR/slurm_logs/${EXP_NAME//\//_}.txt
    fi
done

echo "=================================================="
echo "Finished $MODE execution"
echo "=================================================="




