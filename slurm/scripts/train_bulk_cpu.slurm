#!/bin/bash

#SBATCH --account=group-bradwhitaker       # Priority account to use
#SBATCH --job-name=resnext_train_CPU       # Job name
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
#SBATCH --array=0-3                        # Array job to run 4 tasks (0 to 3)

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

# Extract details from dataset path for descriptive naming
IFS='/' read -r -a path_parts <<< "$DATASET_PATH"
DATASET_NAME=${path_parts[-1]}
BOOST_STATUS=$(echo $DATASET_NAME | grep -q "noboost" && echo "noboost" || echo "boost")
SF_STATUS=$(echo $DATASET_NAME | grep -oP '(all_SF|full_SF|partial_SF|prop-only)' || echo "")
FILL_FACTOR=$(echo $DATASET_NAME | grep -oP '(all|full|partial|prop-only|prop-only-90)' || echo "all")

# Create descriptive work directory name
WORK_DIR="/home/d86p233/Desktop/BMW-spec/work_dirs/train_task_${SLURM_ARRAY_TASK_ID}"

# Define the path to your Apptainer image
IMAGE_PATH="/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif"

# Define the base config path and the script to update the config
BASE_CONFIG_PATH="/home/d86p233/Desktop/BMW-spec/mmdetection/configs/faster_rcnn/faster-rcnn_x101-32x8d_spec.py"
UPDATE_SCRIPT_PATH="/home/d86p233/Desktop/BMW-spec/scripts/update_train_config.py"

# Create the work directory if it doesn't exist
mkdir -p $WORK_DIR

# Create a text file with job details
JOB_DETAILS="${WORK_DIR}/job_details.txt"
echo "Dataset Path: ${DATASET_PATH}" > $JOB_DETAILS
echo "Normalization Method: ${NORMALIZATION_METHOD}" >> $JOB_DETAILS
echo "Boost Status: ${BOOST_STATUS}" >> $JOB_DETAILS
echo "Single Frequency Status: ${SF_STATUS}" >> $JOB_DETAILS
echo "Fill Factor: ${FILL_FACTOR}" >> $JOB_DETAILS

# Run the script to update the config
apptainer exec --nv $IMAGE_PATH python $UPDATE_SCRIPT_PATH $BASE_CONFIG_PATH $DATASET_PATH $WORK_DIR $NORMALIZATION_METHOD

# Run the training with the updated config
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/mmdetection/tools/train.py $WORK_DIR/updated_config.py
