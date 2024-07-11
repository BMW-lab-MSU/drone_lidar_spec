import sys
from mmengine.config import Config
import os

# Get arguments from the command line
config_path = sys.argv[1]
dataset_path = sys.argv[2]
work_dir = sys.argv[3]
gpu_id = int(sys.argv[4])  # Get the GPU ID from the arguments
normalization_method = sys.argv[5]  # Get the normalization method

# Load the config file
config = Config.fromfile(config_path)

# Update dataset paths in the config
config.train_dataloader.dataset.data_root = dataset_path
config.val_dataloader.dataset.data_root = dataset_path
config.test_dataloader.dataset.data_root = dataset_path
config.train_dataloader.dataset.ann_file = f'{dataset_path}/train/annotations.json'
config.val_dataloader.dataset.ann_file = f'{dataset_path}/val/annotations.json'
config.test_dataloader.dataset.ann_file = f'{dataset_path}/test/annotations.json'
config.train_dataloader.dataset.data_prefix.img = f'{dataset_path}/train/'
config.val_dataloader.dataset.data_prefix.img = f'{dataset_path}/val/'
config.test_dataloader.dataset.data_prefix.img = f'{dataset_path}/test/'
config.val_evaluator.ann_file = f'{dataset_path}/val/annotations.json'
config.test_evaluator.ann_file = f'{dataset_path}/test/annotations.json'

# Update the work directory
config.work_dir = work_dir

# Update the GPU ID
config.gpu_ids = [gpu_id]

# Update the normalization method
if normalization_method == 'standard':
    if 'noboost' in dataset_path:
        normalization_values = dict(type='Normalize', mean=[117.72, 201.1, 82.63], std=[55.09, 24.4, 33.32], to_rgb=True)
    else:
        normalization_values = dict(type='Normalize', mean=[91.29, 159.0, 102.64], std=[67.23, 63.18, 38.67], to_rgb=True)
elif normalization_method == 'minmax':
    normalization_values = dict(type='Normalize', mean=[0.0, 0.0, 0.0], std=[1.0/255.0, 1.0/255.0, 1.0/255.0], to_rgb=True)

config.train_pipeline[3] = normalization_values
config.val_pipeline[3] = normalization_values
config.test_pipeline[3] = normalization_values

# Ensure the work directory exists
os.makedirs(work_dir, exist_ok=True)

# Print the final configuration for debugging
print(config.pretty_text)

# Save the updated config
config.dump(f'{work_dir}/updated_config.py')
