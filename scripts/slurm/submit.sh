#!/bin/bash -l

n_devicesS=
N_NODES=
PARTITION=
MEMORY=
CMD=

usage()
{
    echo "Usage:"
    echo "sbatch ./submit.sh --partition|-p=<partition> --memory|-m=<XG> --n_nodes=<n_nodes> --n_devicess=<n_devicess> --cmd=<cmd>"
    echo ""
    echo " -h | --help : Displays the help"
    echo " -p | --partition : GPU partition"
    echo " -m | --memory : Total memory"
    echo " --n_devicess : Number of GPUs per task."
    echo " --n_nodes : Number of nodes."
    echo " --cmd : Command to run. "
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
	    -p  | --partition )
	        PARTITION=$VALUE
	        ;;
	    -m  | --memory )
	        MEMORY=$VALUE
	        ;;
        --n_devicess )
            n_devicesS=$VALUE
            ;;
        --n_nodes)
            N_NODES=$VALUE
            ;;
        --cmd)
            CMD=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

if [ "$PARTITION" = "" ]; then
  usage
  exit 1
fi

if [ "$N_NODES" = "" ]; then
  usage
  exit 1
fi

if [ "$n_devicesS" = "" ]; then
  usage
  exit 1
fi

if [ "$MEMORY" = "" ]; then
  usage
  exit 1
fi

if [ "$CMD" = "" ]; then
  usage
  exit 1
fi

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=$N_NODES
#SBATCH --gres=gpu:$n_devicesS
#SBATCH --ntasks-per-node=$n_devicesS
#SBATCH --mem=$MEMORY
#SBATCH --partition $PARTITION

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# run script from above
srun \
    --container-image=/netscratch/$USER/pytorchlightning+transformers-pytorch-gpu+latest.sqsh \
    --container-workdir=/home/$USER/xai_torch \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/xai_torch:/home/$USER/xai_torch \
    --task-prolog=./scripts/slurm/install.sh \
    -p $PARTITION \
    --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,ROOT_DIR=/,XAI_TORCH_OUTPUT_DIR=/netscratch/$USER/xai_torch,PYTHONPATH=$PYTHONPATH:/home/$USER/xai_torch/src,TORCH_HOME=/netscratch/$USER/xai_torch/pretrained" \
    $CMD
