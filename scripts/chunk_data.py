"""
This script splits the input dataset into smaller chunks to facilitate parallel processing on an HPC system. It copies
the files from the input directory into multiple subdirectories within an output directory. This can be useful for 
submitting multiple jobs to a job scheduler like Slurm to process the data in parallel.

Usage:
    1. Place your input files ('.mat', '.h5', or '.hdf5') in the 'input_files' directory.
    2. Adjust the 'num_splits' argument to the desired number of splits.
    3. Run the script.

    Example:
        python split_data.py --input_folder input_files --output_folder input_splits --num_splits 10

    Arguments:
        --input_folder (str): The path to the input directory containing the files to be split.
        --output_folder (str): The path to the output directory where the split subdirectories will be created.
        --num_splits (int): The number of subdirectories to create. Each subdirectory will contain a portion of the input files.

Output:
    The script creates subdirectories within the specified output directory. Each subdirectory contains a portion of the 
    input files, facilitating parallel processing.

Example Directory Structure:
    input_files/
        file1.mat
        file2.h5
        ...
    input_splits/
        split_0/
            file1.mat
            ...
        split_1/
            file2.h5
            ...
        ...

Notes:
    - Ensure that the 'input_files' directory contains the files to be split.
    - Adjust the 'num_splits' argument according to the number of desired splits.
    - The script can handle files with '.mat', '.h5', and '.hdf5' extensions.
"""

import os
import shutil
from math import ceil
import argparse

def split_data(input_folder, output_folder, num_splits):
    os.makedirs(output_folder, exist_ok=True)
    all_files = [f for f in os.listdir(input_folder) if f.endswith(('.mat', '.h5', '.hdf5'))]
    files_per_split = ceil(len(all_files) / num_splits)

    for i in range(num_splits):
        split_folder = os.path.join(output_folder, f'split_{i}')
        os.makedirs(split_folder, exist_ok=True)
        split_files = all_files[i * files_per_split: (i + 1) * files_per_split]
        for file in split_files:
            shutil.copy(os.path.join(input_folder, file), os.path.join(split_folder, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split input data into smaller chunks for parallel processing.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the input directory containing the files to be split.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output directory where the split subdirectories will be created.")
    parser.add_argument('--num_splits', type=int, required=True, help="The number of subdirectories to create. Each subdirectory will contain a portion of the input files.")

    args = parser.parse_args()
    split_data(args.input_folder, args.output_folder, args.num_splits)
