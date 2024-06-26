#!/bin/bash
#SBATCH --job-name=faster_rcnn_training
#SBATCH --output=faster_rcnn_training.out
#SBATCH --error=faster_rcnn_training.err
#SBATCH --time=4:00:00  # Requesting 4 hours
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2  # Requesting 2 GPUs
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4  # Requesting 4 CPUs (2 workers per GPU)

# Load Apptainer module
module load apptainer

# Define the path to your Apptainer image
IMAGE_PATH=/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif

# Run training script within the Apptainer container
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/mmdetection/tools/train.py /home/d86p233/Desktop/BMW-spec/configs/faster_rcnn_resnext101.py
