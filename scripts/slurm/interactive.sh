srun \
  --container-image=/netscratch/$USER/pytorchlightning+transformers-pytorch-gpu+latest.sqsh \
  --container-workdir=/home/$USER/document_analysis_stack \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/document_analysis_stack:/home/$USER/document_analysis_stack \
  --task-prolog=./scripts/slurm/install.sh \
  --time 01:00:00 --pty /bin/bash
