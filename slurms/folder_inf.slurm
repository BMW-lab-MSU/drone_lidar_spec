#!/bin/bash
##
## gpuexample.sbatch submit a job using a GPU
##
## Lines starting with #SBATCH are read by Slurm. Lines starting with ## are comments.
## All other lines are read by the shell.
##
#SBATCH --account=priority-bradwhitaker # priority account to use
#SBATCH --job-name=mmdet_inf      # job name
#SBATCH --partition=gpupriority         # queue partition to run the job in
#SBATCH --nodes=1                       # number of nodes to allocate
#SBATCH --ntasks-per-node=1             # number of descrete tasks - keep at one except for MPI
#SBATCH --cpus-per-task=8               # number of cores to allocate - do not allocate more than 16 cores per GPU
#SBATCH --gpus-per-task=1               #  number of GPUs to allocate - all GPUs are currently A40 model
#SBATCH --mem=2000                     # 2000 MB of Memory allocated - do not allocate more than 128000 MB mem per GPU
#SBATCH --time=1-00:10:00               # Maximum job run time (d-hh:mm:ss)
#SBATCH --output=slurm_out/image_demo_%j.out  # Store output files in slurm_out directory
#SBATCH --error=slurm_err/image_demo_%j.err   # Store error files in slurm_err directory


## Run 'man sbatch' for more information on the options above.

# Create directories for SLURM output and error files if they do not exist
mkdir -p slurm_out slurm_err

### Replace the below with modules and commands
module load CUDA/11.1.1-GCC-10.2.0 

echo "You are using CUDA version: " 
nvcc --version

# Ensure that the Singularity/Apptainer image and script are in your working directory
# Run the Singularity/Apptainer container with GPU support
apptainer exec --nv mmdet_img.sif python mmdetection/custom_inf/folder_inf.py \
    mmdetection/custom_imgs mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py \
    --weights mmdetection/checkpoints/rtmdet_s_8xb32-300e_coco.pth \
    --out-dir outputs

