#!/bin/bash

#SBATCH --account=group-bradwhitaker       # Priority account to use
#SBATCH --job-name=resnext_test            # Job name
#SBATCH --partition=gpuunsafe              # Queue partition to run the job in
#SBATCH --nodes=1                          # Number of nodes to allocate
#SBATCH --ntasks-per-node=1                # Number of discrete tasks - keep at one except for MPI
#SBATCH --cpus-per-task=2                  # Number of cores to allocate - do not allocate more than 16 cores per GPU
#SBATCH --gpus-per-task=1                  # Number of GPUs to allocate - all GPUs are currently A40 model
#SBATCH --mem=16000                        # Memory allocated - do not allocate more than 128000 MB mem per GPU
#SBATCH --time=11:00:00                    # Maximum job run time (d-hh:mm:ss)
#SBATCH --output=../out/resnext_test_%A_%a.out  # Store output files in slurm_out directory
#SBATCH --error=../err/resnext_test_%A_%a.err   # Store error files in slurm_err directory
#SBATCH --mail-type=BEGIN,END,FAIL         # Notifications for job start, end, and failure
#SBATCH --array=0-1                        # Array job to run 2 tasks (0 to 1)

# Define an array of testing dataset paths
TEST_DATASET_PATHS=(
    "/home/d86p233/Desktop/BMW-spec/testing_data/noboost_prop-only/tilt_90"  # Test dataset for Task/Model 0
    # Add other test dataset paths here if needed
)

# Define normalization methods
NORMALIZATION_METHODS=("standard" "minmax")

# Calculate dataset index and normalization method index
DATASET_INDEX=0  # Assuming only one test dataset
NORM_METHOD_INDEX=$SLURM_ARRAY_TASK_ID

TEST_DATASET_PATH=${TEST_DATASET_PATHS[$DATASET_INDEX]}
NORMALIZATION_METHOD=${NORMALIZATION_METHODS[$NORM_METHOD_INDEX]}

# Define the path to your Apptainer image
IMAGE_PATH=/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif

# Define the base config path and the script to update the config
BASE_CONFIG_PATH=/home/d86p233/Desktop/BMW-spec/mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_spec.py
UPDATE_SCRIPT_PATH=/home/d86p233/Desktop/BMW-spec/scripts/update_test_config.py

# Define a unique work directory for each job
WORK_DIR=/home/d86p233/Desktop/BMW-spec/work_dirs/job_$SLURM_ARRAY_JOB_ID/task_$SLURM_ARRAY_TASK_ID

# Create the work directory if it doesn't exist
mkdir -p $WORK_DIR

# Run the script to update the config with the test dataset path
apptainer exec --nv $IMAGE_PATH python $UPDATE_SCRIPT_PATH $BASE_CONFIG_PATH $TEST_DATASET_PATH $WORK_DIR $SLURM_ARRAY_TASK_ID $NORMALIZATION_METHOD

# Define the specific job id and task id variables
PREV_JOB_ID=2256243
PREV_TASK_ID=0

# Run the testing with the updated config
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/mmdetection/tools/test.py $WORK_DIR/updated_config.py /home/d86p233/Desktop/BMW-spec/work_dirs/task_$PREV_TASK_ID/epoch_12.pth --out $WORK_DIR/results.pkl
