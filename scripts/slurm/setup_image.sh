srun --gpus-per-task=0 \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.08-py3.sqsh \
  --container-workdir=/home/$USER/projects/doc_active_learning \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/:/home/$USER/ \
  --container-save=/netscratch/$USER/envs/xai_torch.sqsh \
  --pty /bin/bash
