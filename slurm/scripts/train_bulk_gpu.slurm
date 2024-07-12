#!/bin/bash

#SBATCH --account=group-bradwhitaker       # Priority account to use
#SBATCH --job-name=resnext_train_GPU       # Job name
#SBATCH --partition=gpuunsafe              # Queue partition to run the job in
#SBATCH --nodes=1                          # Number of nodes to allocate
#SBATCH --ntasks=1                         # Number of discrete tasks
#SBATCH --cpus-per-task=5                  # Number of cores to allocate per task
#SBATCH --gres=gpu:2                       # Number of GPUs to allocate
#SBATCH --mem=16000                        # Memory allocated
#SBATCH --time=11:00:00                    # Maximum job run time (d-hh:mm:ss)
#SBATCH --output=../out/resnext_train_%A_%a.out  # Store output files in slurm_out directory
#SBATCH --error=../err/resnext_train_%A_%a.err   # Store error files in slurm_err directory
#SBATCH --mail-type=BEGIN,END,FAIL         # Notifications for job start, end, and failure
#SBATCH --mail-user=zcrennen@mines.edu     # Email address for notifications
#SBATCH --array=0-3                        # Array job to run tasks (0 to 3)

# Define an array of dataset paths
DATASET_PATHS=(
    "/home/d86p233/Desktop/BMW-spec/data/noboost_prop-only-90"
    "/home/d86p233/Desktop/BMW-spec/data/boost_prop-only-90"
)

# Define normalization methods
NORMALIZATION_METHODS=("standard" "minmax")

# Calculate dataset index and normalization method index
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
NORM_METHOD_INDEX=$((SLURM_ARRAY_TASK_ID % 2))

DATASET_PATH=${DATASET_PATHS[$DATASET_INDEX]}
NORMALIZATION_METHOD=${NORMALIZATION_METHODS[$NORM_METHOD_INDEX]}

# Define the path to your Apptainer image
IMAGE_PATH=/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif

# Define the base config path and the script to update the config
BASE_CONFIG_PATH=/home/d86p233/Desktop/BMW-spec/mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_spec.py
UPDATE_SCRIPT_PATH="/home/d86p233/Desktop/BMW-spec/scripts/update_train_config.py"

# Define a unique work directory for each job
WORK_DIR="/home/d86p233/Desktop/BMW-spec/work_dirs/train_task_${SLURM_ARRAY_TASK_ID}"

# Create the work directory if it doesn't exist
mkdir -p $WORK_DIR

# Print paths for debugging
echo "DATASET_PATH: ${DATASET_PATH}"
echo "WORK_DIR: ${WORK_DIR}"
echo "NORMALIZATION_METHOD: ${NORMALIZATION_METHOD}"

# Load necessary modules (if needed, depends on your HPC environment)
# module load cuda/11.3
# module load singularity

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1

# Run the Python script to update the configuration with the dataset path and normalization method
apptainer exec --nv $IMAGE_PATH python $UPDATE_SCRIPT_PATH $BASE_CONFIG_PATH $DATASET_PATH $WORK_DIR $NORMALIZATION_METHOD

# Check CUDA availability inside the container
apptainer exec --nv $IMAGE_PATH python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print('GPU', i, 'Name:', torch.cuda.get_device_name(i))
"

# Run the training script with the updated configuration within the container
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/mmdetection/tools/train.py $WORK_DIR/updated_config.py --work-dir=$WORK_DIR --launcher="slurm"
