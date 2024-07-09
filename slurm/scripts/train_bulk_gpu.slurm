#!/bin/bash

#SBATCH --account=group-bradwhitaker
#SBATCH --job-name=resnext_train
#SBATCH --partition=gpuunsafe
#SBATCH --nodes=8                # Request 8 nodes
#SBATCH --ntasks-per-node=1      # Request 1 tasks per node
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1        # Request 1 GPU per task
#SBATCH --mem=16000
#SBATCH --time=11:00:00
#SBATCH --output=../out/resnext_train_%A_%a.out
#SBATCH --error=../err/resnext_train_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zcrennen@mines.edu
#SBATCH --array=0-7

DATASET_PATHS=(
    "/home/d86p233/Desktop/BMW-spec/all_specs_sorted_boosted/all"
    "/home/d86p233/Desktop/BMW-spec/all_specs_sorted_boosted/full"
    "/home/d86p233/Desktop/BMW-spec/all_specs_sorted_boosted/partial"
    "/home/d86p233/Desktop/BMW-spec/all_specs_sorted_boosted/prop-only"
    "/home/d86p233/Desktop/BMW-spec/all_specs_sorted/all"
    "/home/d86p233/Desktop/BMW-spec/all_specs_sorted/full"
    "/home/d86p233/Desktop/BMW-spec/all_specs_sorted/partial"
    "/home/d86p233/Desktop/BMW-spec/all_specs_sorted/prop-only"
)

DATASET_PATH=${DATASET_PATHS[$SLURM_ARRAY_TASK_ID]}
IMAGE_PATH=/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif

WORK_DIR="/home/d86p233/Desktop/BMW-spec/slurm/model_work_dirs/dataset_${SLURM_ARRAY_TASK_ID}"

CONFIG_PATH="/home/d86p233/Desktop/BMW-spec/mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_spec.py"

# Print paths for debugging
echo "DATASET_PATH: ${DATASET_PATH}"
echo "WORK_DIR: ${WORK_DIR}"

# Load necessary modules (if needed, depends on your HPC environment)
# module load cuda/11.3
# module load singularity

# Get the assigned GPU ID
GPU_ID=$SLURM_LOCALID

# Run the Python script to update the configuration with the dataset path and GPU ID
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/scripts/update_config.py ${CONFIG_PATH} ${DATASET_PATH} ${WORK_DIR} ${GPU_ID}

# Check CUDA availability inside the container
apptainer exec --nv $IMAGE_PATH python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU Name:', torch.cuda.get_device_name($GPU_ID))
    print('Current CUDA Device:', torch.cuda.current_device())
"

# Run the training script with the updated configuration
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/mmdetection/tools/train.py ${WORK_DIR}/updated_config.py
