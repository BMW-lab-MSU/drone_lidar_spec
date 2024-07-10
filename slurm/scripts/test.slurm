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
#SBATCH --array=0-0                        # Array job to run 16 tasks (0 to 15)

# Define an array of testing dataset paths
TEST_DATASET_PATHS=(
    "/home/d86p233/Desktop/BMW-spec/testing_data/noboost_prop-only/tilt_90"  # Test dataset for Task/Model 0
)

# Define the path to your Apptainer image
IMAGE_PATH=/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif

# Define the base config path and the script to update the config
BASE_CONFIG_PATH=/home/d86p233/Desktop/BMW-spec/mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_spec.py
UPDATE_SCRIPT_PATH=/home/d86p233/Desktop/BMW-spec/scripts/update_test_config.py

# Get the testing dataset path for the current array task
TEST_DATASET_PATH=${TEST_DATASET_PATHS[$SLURM_ARRAY_TASK_ID]}

# Define a unique work directory for each job
WORK_DIR=/home/d86p233/Desktop/BMW-spec/work_dirs/job_$SLURM_ARRAY_JOB_ID/task_$SLURM_ARRAY_TASK_ID

# Run the script to update the config with the test dataset path
apptainer exec --nv $IMAGE_PATH python $UPDATE_SCRIPT_PATH $BASE_CONFIG_PATH $TEST_DATASET_PATH $WORK_DIR $SLURM_ARRAY_TASK_ID

# Run the testing with the updated config
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/mmdetection/tools/test.py $WORK_DIR/updated_config.py /home/d86p233/Desktop/BMW-spec/work_dirs/job_2254338/task_13/epoch_12.pth --out $WORK_DIR/results.pkl
