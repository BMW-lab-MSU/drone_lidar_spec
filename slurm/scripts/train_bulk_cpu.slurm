#!/bin/bash

#SBATCH --account=group-bradwhitaker       # Priority account to use
#SBATCH --job-name=resnext_train           # Job name
#SBATCH --partition=gpuunsafe              # Queue partition to run the job in
#SBATCH --nodes=1                          # Number of nodes to allocate
#SBATCH --ntasks-per-node=1                # Number of discrete tasks - keep at one except for MPI
#SBATCH --cpus-per-task=2                  # Number of cores to allocate - do not allocate more than 16 cores per GPU
#SBATCH --gpus-per-task=1                  # Number of GPUs to allocate - all GPUs are currently A40 model
#SBATCH --mem=16000                        # Memory allocated - do not allocate more than 128000 MB mem per GPU
#SBATCH --time=10:00:00                    # Maximum job run time (d-hh:mm:ss)
#SBATCH --output=../out/resnext_train_%j.out  # Store output files in slurm_out directory
#SBATCH --error=../err/resnext_train_%j.err   # Store error files in slurm_err directory
#SBATCH --mail-type=BEGIN,END,FAIL         # Notifications for job start, end, and failure
#SBATCH --mail-user=zcrennen@mines.edu     # Email address for notifications

# Define an array of dataset paths
DATASET_PATHS=(
    "/home/d86p233/Desktop/BMW-spec/data"
    "/home/d86p233/Desktop/BMW-spec/data/partial"
    "/home/d86p233/Desktop/BMW-spec/data/prop-only_boosted"
    "/home/d86p233/Desktop/BMW-spec/data/full"
    "/home/d86p233/Desktop/BMW-spec/data/prop-only"
    "/home/d86p233/Desktop/BMW-spec/data/all"
    "/home/d86p233/Desktop/BMW-spec/data/all_boosted"
    "/home/d86p233/Desktop/BMW-spec/data/full_boosted"
    "/home/d86p233/Desktop/BMW-spec/data/partial_boosted"
)


# Define the path to your Apptainer image
IMAGE_PATH=/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif

# Load necessary modules (if applicable)
# module load apptainer

# Loop through each dataset path and run the training script
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    echo "Training with dataset at ${DATASET_PATH}"
    apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/mmdetection/tools/train.py \
        /home/d86p233/Desktop/BMW-spec/mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_spec.py \
        --cfg-options data_root=$DATASET_PATH
done
