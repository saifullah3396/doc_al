# Analyzing the Potential of Active Learning for Document Image Classification
This repository contains the code for the paper [Analyzing the Potential of Active Learning for Document Image Classification](To-be-added) by Saifullah Saifullah, Stefan Agne, Andreas Dengel, and Sheraz Ahmed.

Requires Python 3+. For evaluation, please download the data from the links below.

# Installation and Dependencies
Clone the repository along with sub-modules:
```
git clone https://github.com/saifullah3396/doc_al.git --recursive
```

Install the project dependencies.
```
pip install -r requirements.txt
```

Setup the environment variables for running the code.
```
export PYTHONPATH=`pwd`/src:`pwd`/external/xai_torch/src
```
Setup the output directory for generating checkpoints and logs.
```
export XAI_TORCH_OUTPUT_DIR=<path-to-output-dir>
```
Setup the output directory for dataset and models.
```
export XAI_TORCH_CACHE_DIR=<path-to-cache-dir>
```

# Running an experiment
To run an experiment, call the main training script and set args/al_args=<active-learning-config>. For example to run, active learning with entropy sampling on Tobacco3482 dataset, run:
```
./scripts/al_train.sh --config-path ../../../cfg/al_adv +experiment=active_learning/tobacco3482/resnet50 args/al_args=entropy_sampling args.data_args.dataset_dir=<path-to-dataset-dir>
```

# License
This repository is released under the Apache 2.0 license as found in the LICENSE file.

