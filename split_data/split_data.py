import argparse
import os
import shutil
import h5py
import numpy as np
import pandas as pd

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

def create_output_folders(output_dir, prop_combos):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Create train, val, test directories and subdirectories for each prop combo
    for prop in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, prop), exist_ok=True)
        for combo in prop_combos:
            os.makedirs(os.path.join(output_dir, prop, combo), exist_ok=True)

def get_unique_combos(files):
    # Get unique combinations of n_blades and size from the files
    combos = set()
    for file in files:
        with h5py.File(file, 'r') as f:
            n_blades = f.attrs['n_blades']
            size = f.attrs['size']
            combos.add(f'{n_blades}_{size}')
    return combos

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
        args.output_dir = os.path.join(args.data_dir, 'split_data')
    
    # List all HDF5 files in the data directory
    h5_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.h5')]
    
    # Get unique property combinations from the HDF5 files
    prop_combos = get_unique_combos(h5_files)
    # Create output folders for each property combination
    create_output_folders(args.output_dir, prop_combos)
    
    # Process each property combination
    for combo in prop_combos:
        # Get files that match the current property combination
        combo_files = [f for f in h5_files if f'{h5py.File(f, "r").attrs["n_blades"]}_{h5py.File(f, "r").attrs["size"]}' == combo]
        # Split files into train, val, and test sets
        train_files, val_files, test_files = split_data(combo_files, args.train_split, args.val_split, args.test_split)
        
        # Copy files to their respective directories
        for file in train_files:
            shutil.copy(file, os.path.join(args.output_dir, 'train', combo))
        for file in val_files:
            shutil.copy(file, os.path.join(args.output_dir, 'val', combo))
        for file in test_files:
            shutil.copy(file, os.path.join(args.output_dir, 'test', combo))
    
    # Print completion message
    print(f"Data split completed. Output saved in {args.output_dir}")

if __name__ == "__main__":
    main()
