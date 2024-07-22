#!/bin/bash

# Submit the training job array
TRAIN_JOB_ID=$(sbatch --parsable train_bulk_cpu.slurm)

# Submit the testing job array with a dependency on the training job array
sbatch --dependency=afterok:$TRAIN_JOB_ID test_bulk_cpu.slurm
