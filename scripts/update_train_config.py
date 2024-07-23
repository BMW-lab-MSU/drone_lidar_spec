"""
This script updates a given MMDetection configuration file with specified dataset paths, work directory, and normalization method.
It modifies the configuration file to ensure proper dataset paths, work directory, and normalization values are set, and removes any 
specific GPU settings to maintain flexibility for different training environments.

Usage:
    python update_train_config.py <config_path> <dataset_path> <work_dir> <normalization_method>

Arguments:
    config_path (str): Path to the MMDetection configuration file.
    dataset_path (str): Path to the dataset directory. The script expects subdirectories 'train', 'val', and 'test' 
                        within this path, each containing the respective images and an 'annotations.json' file.
    work_dir (str): Path to the directory where the updated configuration file and training outputs will be saved.
    normalization_method (str): The normalization method to be used. It can be 'standard' or 'minmax'. 
                                - 'standard': Uses pre-defined mean and standard deviation values based on the dataset path.
                                - 'minmax': Uses mean [0.0, 0.0, 0.0] and standard deviation [255.0, 255.0, 255.0] for min-max normalization.

Example:
    python update_train_config.py configs/faster_rcnn_r50_fpn.py /path/to/dataset /path/to/work_dir standard

Details:
    - The script loads the specified configuration file and updates the dataset paths to point to the provided dataset directory.
    - It updates the work directory in the configuration to the provided path and ensures it exists.
    - The normalization method is applied to update the mean and standard deviation values in the configuration's data preprocessor 
      and dataset pipelines.
    - The final configuration is printed for debugging and saved to the specified work directory as 'updated_config.py'.

Note:
    - Ensure that the dataset directory contains the 'train', 'val', and 'test' subdirectories, each with an 'annotations.json' file 
      and the respective images.
    - This script is designed for use with MMDetection configuration files.
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

# Print initial configuration for debugging
print("Initial train_dataloader.dataset configuration:")
print(config.train_dataloader.dataset)
print("Initial val_dataloader.dataset configuration:")
print(config.val_dataloader.dataset)
print("Initial test_dataloader.dataset configuration:")
print(config.test_dataloader.dataset)

# Access nested dataset for train_dataloader if it's a RepeatDataset
train_dataset = config.train_dataloader.dataset
if train_dataset.type == 'RepeatDataset':
    train_dataset = train_dataset.dataset

# Update dataset paths in the config using os.path.join to avoid double slashes
config.data_root = dataset_path
train_dataset.data_root = dataset_path
config.val_dataloader.dataset.data_root = dataset_path
config.test_dataloader.dataset.data_root = dataset_path

train_dataset.ann_file = os.path.join(dataset_path, 'train', 'annotations.json')
config.val_dataloader.dataset.ann_file = os.path.join(dataset_path, 'val', 'annotations.json')
config.test_dataloader.dataset.ann_file = os.path.join(dataset_path, 'test', 'annotations.json')

# Check and update data_prefix if it exists within the dataset dictionary
if 'data_prefix' in train_dataset:
    print("FOUND TRAIN DATA PREFIX")
    train_dataset.data_prefix['img'] = os.path.join(dataset_path, 'train')
else:
    print("TRAIN DATA PREFIX NOT FOUND")

if 'data_prefix' in config.val_dataloader.dataset:
    print("FOUND VAL DATA PREFIX")
    config.val_dataloader.dataset.data_prefix['img'] = os.path.join(dataset_path, 'val')
else:
    print("VAL DATA PREFIX NOT FOUND")

if 'data_prefix' in config.test_dataloader.dataset:
    print("FOUND TEST DATA PREFIX")
    config.test_dataloader.dataset.data_prefix['img'] = os.path.join(dataset_path, 'test')
else:
    print("TEST DATA PREFIX NOT FOUND")

config.val_evaluator.ann_file = os.path.join(dataset_path, 'val', 'annotations.json')
config.test_evaluator.ann_file = os.path.join(dataset_path, 'test', 'annotations.json')

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
for pipeline in [train_dataset.pipeline, config.val_dataloader.dataset.pipeline, config.test_dataloader.dataset.pipeline]:
    for step in pipeline:
        if step['type'] == 'Normalize':
            step.update(norm_values_pipeline)

# Update normalization values in the config directly
config.normalization_values = normalization_values

# Ensure the work directory exists
os.makedirs(work_dir, exist_ok=True)

# Print the final configuration for debugging
print("Updated train_dataloader.dataset configuration:")
print(config.train_dataloader.dataset)
print("Updated val_dataloader.dataset configuration:")
print(config.val_dataloader.dataset)
print("Updated test_dataloader.dataset configuration:")
print(config.test_dataloader.dataset)

print(config.pretty_text)

# Save the updated config
config.dump(os.path.join(work_dir, 'updated_config.py'))
