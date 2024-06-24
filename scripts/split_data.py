"""
This script is designed to split a dataset of HDF5 files into training, validation, and test sets. It takes in a directory
containing the raw data files and splits them according to specified percentages. The split files are saved in an output 
directory, which can be specified by the user or will default to a directory named 'splits' in the same parent directory 
as the input data.

Usage:
    python split_data.py --data_dir /path/to/raw/data --output_dir /path/to/output/folder --train_split 70 --val_split 15 --test_split 15

Arguments:
    --data_dir:       (Required) Directory where raw data is stored.
    --output_dir:     (Optional) Output directory (create one if left blank). Defaults to 'splits' in the parent directory of data_dir.
    --train_split:    (Required) Percentage of data for training set.
    --val_split:      (Optional) Percentage of data for validation set. Defaults to 0 if not provided.
    --test_split:     (Required) Percentage of data for test set.

Note:
    The sum of train_split, val_split, and test_split must equal 100.
"""

# Optional todo: modify this to support cross fold validation and split correctly

import argparse
import os
import shutil
import h5py
import numpy as np
import logging

def parse_args():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Split data into train/val/test sets.')
    parser.add_argument('--data_dir', required=True, help='Directory where raw data is stored')
    parser.add_argument('--output_dir', default=None, help='Output directory (create one if left blank)')
    parser.add_argument('--train_split', type=int, required=True, help='Percentage of data for training set')
    parser.add_argument('--val_split', type=int, default=0, help='Percentage of data for validation set')
    parser.add_argument('--test_split', type=int, required=True, help='Percentage of data for test set')
    return parser.parse_args()

def validate_splits(train_split, val_split, test_split):
    # Ensure the sum of splits equals 100
    total = train_split + val_split + test_split
    if total != 100:
        raise ValueError('Sum of train, val, and test splits must be 100')

def create_output_folders(output_dir):
    # Delete the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # Create output directory
    os.makedirs(output_dir)
    # Create train, val, test directories
    for prop in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, prop), exist_ok=True)

def get_unique_combos(files):
    # Get unique combinations of n_blades and size from the files
    combos = set()
    for file in files:
        with h5py.File(file, 'r') as f:
            n_blades = int(f['parameters/n_blades'][()])
            size = str(f['parameters/prop_size'][()], 'utf-8')
            combos.add(f'{n_blades}_{size}')
    return combos

def filter_files_by_combo(files, combo):
    # Filter files by their property combinations
    filtered_files = []
    for file in files:
        with h5py.File(file, 'r') as f:
            n_blades = int(f['parameters/n_blades'][()])
            size = str(f['parameters/prop_size'][()], 'utf-8')
            if f'{n_blades}_{size}' == combo:
                filtered_files.append(file)
    return filtered_files

def split_data(files, train_split, val_split, test_split):
    # Randomly shuffle and split the files according to the specified splits
    np.random.shuffle(files)
    train_end = int(len(files) * (train_split / 100))
    val_end = train_end + int(len(files) * (val_split / 100))
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    return train_files, val_files, test_files

def main():
    # Parse command-line arguments
    args = parse_args()
    # Validate the split percentages
    validate_splits(args.train_split, args.val_split, args.test_split)
    
    # Set output directory to a default if not provided
    if args.output_dir is None:
        base_output_dir = os.path.dirname(os.path.abspath(args.data_dir))
        args.output_dir = os.path.join(base_output_dir, 'splits')
    
    # List all HDF5 files in the data directory
    h5_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.hdf5')]
    
    # Get unique property combinations from the HDF5 files
    prop_combos = get_unique_combos(h5_files)
    # Create output folders for each property combination
    create_output_folders(args.output_dir)
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Initialize file counters
    train_counter = 0
    val_counter = 0
    test_counter = 0
    
    # Process each property combination
    for combo in prop_combos:
        # Get files that match the current property combination
        combo_files = filter_files_by_combo(h5_files, combo)
        # Split files into train, val, and test sets
        train_files, val_files, test_files = split_data(combo_files, args.train_split, args.val_split, args.test_split)
        
        # Copy files to their respective directories and log progress
        for file in train_files:
            shutil.copy(file, os.path.join(args.output_dir, 'train'))
            train_counter += 1
            if train_counter % 50 == 0:
                logging.info(f'{train_counter} files copied to train folder')

        for file in val_files:
            shutil.copy(file, os.path.join(args.output_dir, 'val'))
            val_counter += 1
            if val_counter % 50 == 0:
                logging.info(f'{val_counter} files copied to val folder')

        for file in test_files:
            shutil.copy(file, os.path.join(args.output_dir, 'test'))
            test_counter += 1
            if test_counter % 50 == 0:
                logging.info(f'{test_counter} files copied to test folder')
    
    # Print completion message
    logging.info(f"Data split completed. Output saved in {args.output_dir}")

if __name__ == "__main__":
    main()
