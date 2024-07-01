#!/bin/bash

#SBATCH --account=group-bradwhitaker # priority account to use
#SBATCH --job-name=resnext_train      # job name
#SBATCH --partition=gpuunsafe         # queue partition to run the job in
#SBATCH --nodes=1                       # number of nodes to allocate
#SBATCH --ntasks-per-node=1             # number of descrete tasks - keep at one except for MPI
#SBATCH --cpus-per-task=4               # number of cores to allocate - do not allocate more than 16 cores per GPU
#SBATCH --gpus-per-task=2               #  number of GPUs to allocate - all GPUs are currently A40 model
#SBATCH --mem=32000                     # 2000 MB of Memory allocated - do not allocate more than 128000 MB mem per GPU
#SBATCH --time=04:00:00               # Maximum job run time (d-hh:mm:ss)
#SBATCH --output=../out/resnext_train_%j.out  # Store output files in slurm_out directory
#SBATCH --error=../err/resnext_train_%j.err   # Store error files in slurm_err directory


# Define the path to your Apptainer image
IMAGE_PATH=/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif

# Run training script within the Apptainer container
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/mmdetection/tools/train.py /home/d86p233/Desktop/BMW-spec/configs/base_config.py
