#!/bin/bash

#SBATCH --account=group-bradwhitaker       # Priority account to use
#SBATCH --job-name=slide_window            # Job name
#SBATCH --partition=gpuunsafe              # Queue partition to run the job in
#SBATCH --nodes=1                          # Number of nodes to allocate
#SBATCH --ntasks-per-node=1                # Number of discrete tasks - keep at one except for MPI
#SBATCH --cpus-per-task=2                  # Number of cores to allocate - do not allocate more than 16 cores per GPU
#SBATCH --mem=16000                        # Memory allocated - do not allocate more than 128000 MB mem per GPU
#SBATCH --time=11:00:00                    # Maximum job run time (d-hh:mm:ss)
#SBATCH --output=/home/d86p233/Desktop/BMW-spec/work_dirs/slurm_out/%x_%j.out  # Store output files in slurm_out directory
#SBATCH --error=/home/d86p233/Desktop/BMW-spec/work_dirs/slurm_err/%x_%j.err   # Store error files in slurm_err directory
#SBATCH --mail-type=BEGIN,END,FAIL         # Notifications for job start, end, and failure
#SBATCH --mail-user=zcrennen@mines.edu     # Email address for notifications

# Create work directories if they don't exist
mkdir -p /home/d86p233/Desktop/BMW-spec/work_dirs/slurm_out
mkdir -p /home/d86p233/Desktop/BMW-spec/work_dirs/slurm_err

# Variables
WORK_DIR="/home/d86p233/Desktop/BMW-spec/work_dirs/SLIDE_noboost_prop_only"
IMAGES_PATH="/home/d86p233/Desktop/BMW-spec/merged_data/noboost_prop-only/images"
OUTPUT_PATH="$WORK_DIR/noboost_prop-only.json"
H5_JSON="SFh5s.json"
IMAGE_JSON="$OUTPUT_PATH"
COMPARISON_OUTPUT="$WORK_DIR/noboost_prop-only-comparisons.json"
METRICS_JSON="$COMPARISON_OUTPUT"
RESULTS_JSON="$WORK_DIR/noboost_prop-only-results.json"
IMAGE_PATH="/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif"

# Create the specific work directory
mkdir -p "$WORK_DIR"

# Log start time
echo "Starting job at $(date)"

# Run the sliding window script
echo "Running slide_windows.py at $(date)"
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/sliding_window/slide_windows.py --images_path "$IMAGES_PATH" --output_path "$OUTPUT_PATH" > "$WORK_DIR/slide_windows.log" 2>&1

# Run the frequency comparison script
echo "Running compare_frequencies.py at $(date)"
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/sliding_window/compare_frequencies.py --h5_json "$H5_JSON" --image_json "$IMAGE_JSON" --output_path "$COMPARISON_OUTPUT" > "$WORK_DIR/compare_frequencies.log" 2>&1

# Run the evaluation metrics script
echo "Running calc_eval_metrics.py at $(date)"
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/sliding_window/calc_eval_metrics.py --json_path "$METRICS_JSON" --output_path "$RESULTS_JSON" > "$WORK_DIR/calc_eval_metrics.log" 2>&1

# Log end time
echo "Job finished at $(date)"
