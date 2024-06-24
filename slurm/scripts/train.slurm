#!/bin/bash
#SBATCH --job-name=faster_rcnn_training
#SBATCH --output=faster_rcnn_training.out
#SBATCH --error=faster_rcnn_training.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load Apptainer module
module load apptainer

# Path to your Apptainer image
APPTAINER_IMAGE=/path/to/your/image.sif

# Run the training script inside the Apptainer container
apptainer exec --nv $APPTAINER_IMAGE \
    bash -c "source /path/to/your/venv/bin/activate && \
    cd /path/to/your/mmdetection && \
    python tools/train.py configs/your_project/main_config.py --work-dir /path/to/your/work_dir"
