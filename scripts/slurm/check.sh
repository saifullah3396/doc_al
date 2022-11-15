#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

srun hostname