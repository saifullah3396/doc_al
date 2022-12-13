# Analyzing the Potential of Active Learning for Document Image Classification
This repository contains the code for the paper [Analyzing the Potential of Active Learning for Document Image Classification](https://www.researchsquare.com/article/rs-2273654/v1) by Saifullah Saifullah, Stefan Agne, Andreas Dengel, and Sheraz Ahmed.

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

# Running an experiment directly
To run an experiment, call the main training script and set args/al_args=<active-learning-config>. For example to run, active learning with entropy sampling on Tobacco3482 dataset, run:
```
./scripts/al_train.sh --config-path ../../../cfg/al_adv +experiment=active_learning/tobacco3482/resnet50 args/al_args=entropy_sampling args.data_args.dataset_dir=<path-to-dataset-dir>
```

# Running experiments through a helper script
To run different experiments, the helper scripts ./experiments/tobacco3482.sh, ./experiments/tobacco3482_pre.sh, ./experiments/rvlcdip.sh can be used. For example to run an experiment with tobacco3482 dataset, class imbalance m=2, and entropy sampling, the following script can be used.
```
./scripts/experiments/tobacco3482.sh --query=entropy_sampling --exp=imb_2 --data-path=<path-to-dataset-dir> --seed=0
```

# Citation
If you find this paper helpful in your work, please consider citing:
```
Saifullah Saifullah, Stefan Agne, Andreas Dengel et al. Analyzing the Potential of Active Learning for Document Image Classification, 22 November 2022, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-2273654/v1]
```

# License
This repository is released under the Apache 2.0 license as found in the LICENSE file.

