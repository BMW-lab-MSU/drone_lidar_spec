#!/bin/bash

#SBATCH --account=group-bradwhitaker    # priority account to use
#SBATCH --job-name=gen_spec             # job name
#SBATCH --partition=gpuunsafe           # queue partition to run the job in
#SBATCH --nodes=1                       # number of nodes to allocate
#SBATCH --ntasks-per-node=1             # number of discrete tasks - keep at one except for MPI
#SBATCH --cpus-per-task=2               # number of cores to allocate - do not allocate more than 16 cores per GPU
#SBATCH --gpus-per-task=1               # number of GPUs to allocate - all GPUs are currently A40 model
#SBATCH --mem=16000                     # Memory allocated - do not allocate more than 128000 MB mem per GPU
#SBATCH --time=05:00:00                 # Maximum job run time (d-hh:mm:ss)
#SBATCH --output=../gen_spec_out/gen_spec_%A_%a.out  # Store output files in gen_spec_out directory
#SBATCH --error=../gen_spec_err/gen_spec_%A_%a.err   # Store error files in gen_spec_err directory
#SBATCH --array=0-7                     # Adjust this range according to the number of splits

# DATA CAN BE SPLIT WITH scripts/chunk_data.py

# Define the path to your Apptainer image
IMAGE_PATH=/home/d86p233/Desktop/BMW-spec/bmw_spec_img.sif

# Define the input and output folders for the current task
INPUT_FOLDER="/home/d86p233/Desktop/BMW-spec/all_h5s_chunked/split_${SLURM_ARRAY_TASK_ID}"
OUTPUT_FOLDER="/home/d86p233/Desktop/BMW-spec/all_spectrograms_chunked_noBoost/split_${SLURM_ARRAY_TASK_ID}"

# Run the Python script within the Apptainer container
apptainer exec --nv $IMAGE_PATH python /home/d86p233/Desktop/BMW-spec/specs/gen_spec.py $INPUT_FOLDER --output_folder $OUTPUT_FOLDER --range_bins 121 --filter_order 1 --time_slices 32
