"""
Script to update and modify MMDetection configuration files for object detection tasks.

This script loads a configuration file, updates dataset paths, normalization methods, 
and other parameters based on command line arguments, and saves the updated configuration.

Usage:
    python update_train_config.py <config_path> <dataset_path> <work_dir> <normalization_method>

Arguments:
    config_path: Path to the configuration file to be modified.
    dataset_path: Root path to the dataset containing train, val, and test splits.
    work_dir: Directory to save the updated configuration file and any output files.
    normalization_method: Method to use for data normalization ('standard' or 'minmax').

Example:
    python update_train_config.py configs/my_config.py /data/my_dataset /output standard
"""

import sys
from mmengine.config import Config
import os

# Get arguments from the command line
config_path = sys.argv[1]
dataset_path = sys.argv[2]
work_dir = sys.argv[3]
normalization_method = sys.argv[4]  # Get the normalization method

# Load the config file
config = Config.fromfile(config_path)

# Function to update dataset paths
def update_dataset_paths(dataloader, split):
    # Navigate to the inner dataset if it exists
    if 'dataset' in dataloader:
        dataset = dataloader.dataset
        while 'dataset' in dataset:
            dataset = dataset.dataset
    else:
        dataset = dataloader
    dataset.data_root = dataset_path
    dataset.ann_file = os.path.join(dataset_path, f"{split}/annotations/annotations.json")
    if 'data_prefix' in dataset:
        dataset.data_prefix['img'] = os.path.join(dataset_path, f"{split}/images/")
    else:
        dataset.data_prefix = dict(img=os.path.join(dataset_path, f"{split}/images/"))

# Update dataset paths in the config
update_dataset_paths(config.train_dataloader, 'train')
update_dataset_paths(config.val_dataloader, 'val')
update_dataset_paths(config.test_dataloader, 'test')

# Update the work directory
config.work_dir = work_dir

# Remove any GPU settings to ensure flexibility for single or multiple GPUs
if hasattr(config, 'gpu_ids'):
    del config.gpu_ids

# Ensure distributed training is properly configured
config.dist_params = dict(backend='nccl')
config.launcher = 'slurm'

# Define normalization values based on the dataset path and method
if normalization_method == 'standard':
    if 'noboost' in dataset_path:
        mean = [117.72, 201.1, 82.63]
        std = [55.09, 24.4, 33.32]
    else:
        mean = [91.29, 159.0, 102.64]
        std = [67.23, 63.18, 38.67]
elif normalization_method == 'minmax':
    mean = [0.0, 0.0, 0.0]
    std = [255.0, 255.0, 255.0]

# Update normalization values
normalization_values = {
    'mean': mean,
    'std': std
}

# Update normalization in the data preprocessor
config.model.data_preprocessor.mean = mean
config.model.data_preprocessor.std = std

# Update normalization in the training, validation, and test pipelines
norm_values_pipeline = dict(type='Normalize', mean=mean, std=std, to_rgb=True)

def update_pipeline(pipeline):
    for step in pipeline:
        if step['type'] == 'Normalize':
            step.update(norm_values_pipeline)

for dataloader in [config.train_dataloader, config.val_dataloader, config.test_dataloader]:
    # Navigate to the inner dataset if it exists and update its pipeline
    if 'dataset' in dataloader and 'pipeline' in dataloader.dataset:
        dataset = dataloader.dataset
        while 'dataset' in dataset:
            dataset = dataset.dataset
        update_pipeline(dataset.pipeline)
    elif 'pipeline' in dataloader:
        update_pipeline(dataloader.pipeline)

# Update normalization values in the config directly
config.normalization_values = normalization_values

# Ensure the work directory exists
os.makedirs(work_dir, exist_ok=True)

# Print the final configuration for debugging
print(config.pretty_text)

# Save the updated config
config.dump(os.path.join(work_dir, 'updated_config.py'))
