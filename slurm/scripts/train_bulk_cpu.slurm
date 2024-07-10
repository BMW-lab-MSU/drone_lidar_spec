#!/bin/bash

#SBATCH --account=group-bradwhitaker       # Priority account to use
#SBATCH --job-name=resnext_train           # Job name
#SBATCH --partition=gpuunsafe              # Queue partition to run the job in
#SBATCH --nodes=1                          # Number of nodes to allocate
#SBATCH --ntasks-per-node=1                # Number of discrete tasks - keep at one except for MPI
#SBATCH --cpus-per-task=2                  # Number of cores to allocate - do not allocate more than 16 cores per GPU
#SBATCH --gpus-per-task=1                  # Number of GPUs to allocate - all GPUs are currently A40 model
#SBATCH --mem=16000                        # Memory allocated - do not allocate more than 128000 MB mem per GPU
#SBATCH --time=11:00:00                    # Maximum job run time (d-hh:mm:ss)
#SBATCH --output=../out/resnext_train_%A_%a.out  # Store output files in slurm_out directory
#SBATCH --error=../err/resnext_train_%A_%a.err   # Store error files in slurm_err directory
#SBATCH --mail-type=BEGIN,END,FAIL         # Notifications for job start, end, and failure
#SBATCH --mail-user=zcrennen@mines.edu     # Email address for notifications
#SBATCH --array=0-15                        # Array job to run 16 tasks (0 to 15)

# Define an array of dataset paths
DATASET_PATHS=(
    "/home/d86p233/Desktop/BMW-spec/data/all"                  # Task/Model 0: All data
    "/home/d86p233/Desktop/BMW-spec/data/all_SF"               # Task/Model 1: All data with Single Frequency
    "/home/d86p233/Desktop/BMW-spec/data/all_boosted"          # Task/Model 2: All data boosted
    "/home/d86p233/Desktop/BMW-spec/data/all_boosted_SF"       # Task/Model 3: All data boosted with Single Frequency
    "/home/d86p233/Desktop/BMW-spec/data/full"                 # Task/Model 4: Full data
    "/home/d86p233/Desktop/BMW-spec/data/full_SF"              # Task/Model 5: Full data with Single Frequency
    "/home/d86p233/Desktop/BMW-spec/data/full_boosted"         # Task/Model 6: Full data boosted
    "/home/d86p233/Desktop/BMW-spec/data/full_boosted_SF"      # Task/Model 7: Full data boosted with Single Frequency
    "/home/d86p233/Desktop/BMW-spec/data/partial"              # Task/Model 8: Partial data
    "/home/d86p233/Desktop/BMW-spec/data/partial_SF"           # Task/Model 9: Partial data with Single Frequency
    "/home/d86p233/Desktop/BMW-spec/data/partial_boosted"      # Task/Model 10: Partial data boosted
    "/home/d86p233/Desktop/BMW-spec/data/partial_boosted_SF"   # Task/Model 11: Partial data boosted with Single Frequency
    "/home/d86p233/Desktop/BMW-spec/data/prop-only"            # Task/Model 12: Propeller-only data
    "/home/d86p233/Desktop/BMW-spec/data/prop-only-90"         # Task/Model 13: Propeller-only data at 90 degrees
    "/home/d86p233/Desktop/BMW-spec/data/prop-only-90_boosted" # Task/Model 14: Propeller-only data at 90 degrees boosted
    "/home/d86p233/Desktop/BMW-spec/data/prop-only_boosted"    # Task/Model 15: Propeller-only data boosted
)



# Define the path to your Apptainer image
IMAGE_PATH=/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif

# Define the base config path and the script to update the config
BASE_CONFIG_PATH=/home/d86p233/Desktop/BMW-spec/mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_spec.py
UPDATE_SCRIPT_PATH=/home/d86p233/Desktop/BMW-spec/scripts/update_train_config.py

# Get the dataset path for the current array task
DATASET_PATH=${DATASET_PATHS[$SLURM_ARRAY_TASK_ID]}

# Define a unique work directory for each job
WORK_DIR=/home/d86p233/Desktop/BMW-spec/work_dirs/job_$SLURM_ARRAY_JOB_ID/task_$SLURM_ARRAY_TASK_ID

# Run the script to update the config
apptainer exec --nv $IMAGE_PATH python $UPDATE_SCRIPT_PATH $BASE_CONFIG_PATH $DATASET_PATH $WORK_DIR $SLURM_ARRAY_TASK_ID

# Run the training with the updated config
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/mmdetection/tools/train.py $WORK_DIR/updated_config.py
