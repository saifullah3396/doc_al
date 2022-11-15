#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [ -d "/netscratch/$USER_DIR/envs/xai_torch" ]; then
    source /netscratch/$USER_DIR/envs/xai_torch/bin/activate
fi
LOG_LEVEL=INFO python3 $SCRIPT_DIR/../src/al/trainers/train.py $@


