#!/bin/bash

# Define an array of testing dataset paths
TEST_DATASET_PATHS=(
    "/home/d86p233/Desktop/BMW-spec/testing_data/noboost_prop-only/tilt_90"  # Test dataset for Task/Model 0
)

# Define the path to your Apptainer image
IMAGE_PATH=/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif

# Define the base config path and the script to update the config
BASE_CONFIG_PATH=/home/d86p233/Desktop/BMW-spec/mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_spec.py
UPDATE_SCRIPT_PATH=/home/d86p233/Desktop/BMW-spec/scripts/update_test_config.py

# Get the testing dataset path for the current task (assuming single task here for simplicity)
TEST_DATASET_PATH=${TEST_DATASET_PATHS[0]}

# Define a unique work directory for the job
WORK_DIR=/home/d86p233/Desktop/BMW-spec/work_dirs/job_2256243/task_0

# Run the script to update the config with the test dataset path
apptainer exec --nv $IMAGE_PATH python $UPDATE_SCRIPT_PATH $BASE_CONFIG_PATH $TEST_DATASET_PATH $WORK_DIR 0

# Define the specific job id and task id variables
PREV_JOB_ID=2256243
PREV_TASK_ID=0

# Run the testing with the updated config
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/mmdetection/tools/test.py $WORK_DIR/updated_config.py /home/d86p233/Desktop/BMW-spec/work_dirs/task_$PREV_TASK_ID/epoch_12.pth --out $WORK_DIR/results.pkl
