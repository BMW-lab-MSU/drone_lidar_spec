#!/bin/bash

#SBATCH --account=group-bradwhitaker       # Priority account to use
#SBATCH --job-name=slide_window            # Job name
#SBATCH --partition=gpuunsafe              # Queue partition to run the job in
#SBATCH --nodes=1                          # Number of nodes to allocate
#SBATCH --ntasks-per-node=1                # Number of discrete tasks - keep at one except for MPI
#SBATCH --cpus-per-task=2                  # Number of cores to allocate - do not allocate more than 16 cores per GPU
#SBATCH --mem=16000                        # Memory allocated - do not allocate more than 128000 MB mem per GPU
#SBATCH --time=11:00:00                    # Maximum job run time (d-hh:mm:ss)
#SBATCH --output=/home/d86p233/Desktop/BMW-spec/slurm/out/%x_%j.out  # Store output files in slurm_out directory
#SBATCH --error=/home/d86p233/Desktop/BMW-spec/slurm/err/%x_%j.err   # Store error files in slurm_err directory
#SBATCH --mail-type=BEGIN,END,FAIL         # Notifications for job start, end, and failure
#SBATCH --mail-user=zcrennen@mines.edu     # Email address for notifications

# Create work directories if they don't exist
mkdir -p /home/d86p233/Desktop/BMW-spec/slurm/out
mkdir -p /home/d86p233/Desktop/BMW-spec/slurm/err

# Variables
WORK_DIR="/home/d86p233/Desktop/BMW-spec/work_dirs/SLIDE_noboost_prop_only_DEBUG"
IMAGES_PATH="/home/d86p233/Desktop/BMW-spec/data/noboost_prop-only-90/val/images"
OUTPUT_PATH="$WORK_DIR/noboost_prop-only.json"
H5_JSON="/home/d86p233/Desktop/BMW-spec/sliding_window/SFh5s.json"
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
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/sliding_window/slide_windows.py --images_path "$IMAGES_PATH" --output_path "$OUTPUT_PATH"
if [ $? -eq 0 ]; then
    echo "slide_windows.py completed successfully at $(date)"
else
    echo "slide_windows.py failed at $(date)"
    exit 1
fi

# Verify output of slide_windows.py
if [ -f "$OUTPUT_PATH" ]; then
    echo "Output of slide_windows.py exists: $OUTPUT_PATH"
else
    echo "Output of slide_windows.py does not exist: $OUTPUT_PATH"
    exit 1
fi

# Run the frequency comparison script
echo "Running compare_frequencies.py at $(date)"
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/sliding_window/compare_frequencies.py --h5_json "$H5_JSON" --image_json "$IMAGE_JSON" --output_path "$COMPARISON_OUTPUT"
if [ $? -eq 0 ]; then
    echo "compare_frequencies.py completed successfully at $(date)"
else
    echo "compare_frequencies.py failed at $(date)"
    exit 1
fi

# Verify output of compare_frequencies.py
if [ -f "$COMPARISON_OUTPUT" ]; then
    echo "Output of compare_frequencies.py exists: $COMPARISON_OUTPUT"
else
    echo "Output of compare_frequencies.py does not exist: $COMPARISON_OUTPUT"
    exit 1
fi

# Run the evaluation metrics script
echo "Running calc_eval_metrics.py at $(date)"
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/sliding_window/calc_eval_metrics.py --json_path "$METRICS_JSON" --output_path "$RESULTS_JSON"
if [ $? -eq 0 ]; then
    echo "calc_eval_metrics.py completed successfully at $(date)"
else
    echo "calc_eval_metrics.py failed at $(date)"
    exit 1
fi

# Verify output of calc_eval_metrics.py
if [ -f "$RESULTS_JSON" ]; then
    echo "Output of calc_eval_metrics.py exists: $RESULTS_JSON"
else
    echo "Output of calc_eval_metrics.py does not exist: $RESULTS_JSON"
    exit 1
fi

# Log end time
echo "Job finished at $(date)"
