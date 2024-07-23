import sys
from mmengine.config import Config
import os
import json
import numpy as np
from mmengine.registry import DATASETS
from mmdet.datasets import CocoDataset

# Register the CocoDataset with the registry
DATASETS.register_module(CocoDataset)

def check_annotations(dataset_path, phase):
    annotations_file = os.path.join(dataset_path, phase, 'annotations.json')
    if not os.path.exists(annotations_file):
        print(f"Error: {phase} annotations file not found at {annotations_file}")
        return False

    with open(annotations_file, 'r') as f:
        data = json.load(f)

    num_images = len(data.get('images', []))
    num_annotations = len(data.get('annotations', []))

    if num_images == 0 or num_annotations == 0:
        print(f"Error: {phase} dataset is empty. Found {num_images} images and {num_annotations} annotations.")
        return False
    else:
        print(f"Success: {phase} dataset contains {num_images} images and {num_annotations} annotations.")
        return True

def debug_dataset_loader(config, phase):
    try:
        dataset_cfg = getattr(config.data, phase)
        dataset = DATASETS.build(dataset_cfg)
        print(f"Loaded {phase} dataset with {len(dataset)} items.")
        for i, item in enumerate(dataset):
            print(f"{phase} dataset item {i}: {item}")
            if i >= 5:  # Print only first 5 items for brevity
                break
    except Exception as e:
        print(f"Error loading {phase} dataset: {e}")

# Get arguments from the command line
config_path = sys.argv[1]
dataset_path = sys.argv[2]
work_dir = sys.argv[3]
normalization_method = sys.argv[4]  # Get the normalization method

# Load the config file
config = Config.fromfile(config_path)

# Check datasets and update paths in the config
for phase in ['train', 'val', 'test']:
    if not check_annotations(dataset_path, phase):
        sys.exit(f"Invalid {phase} dataset. Exiting...")

    if hasattr(config.data, phase):
        phase_data = getattr(config.data, phase)
        phase_data.data_root = dataset_path
        phase_data.ann_file = os.path.join(dataset_path, phase, 'annotations.json')
        phase_data.img_prefix = os.path.join(dataset_path, phase)
        print(f"Updated {phase} dataset paths:")
        print(f"  data_root: {phase_data.data_root}")
        print(f"  ann_file: {phase_data.ann_file}")
        print(f"  img_prefix: {phase_data.img_prefix}")

if hasattr(config, 'data_root'):
    config.data_root = dataset_path
    print(f"Updated top-level data_root: {config.data_root}")

# Update dataloaders
config.train_dataloader.dataset.data_root = dataset_path
config.val_dataloader.dataset.data_root = dataset_path
config.test_dataloader.dataset.data_root = dataset_path
config.train_dataloader.dataset.ann_file = os.path.join(dataset_path, 'train', 'annotations.json')
config.val_dataloader.dataset.ann_file = os.path.join(dataset_path, 'val', 'annotations.json')
config.test_dataloader.dataset.ann_file = os.path.join(dataset_path, 'test', 'annotations.json')
config.train_dataloader.dataset.data_prefix.img = os.path.join(dataset_path, 'train')
config.val_dataloader.dataset.data_prefix.img = os.path.join(dataset_path, 'val')
config.test_dataloader.dataset.data_prefix.img = os.path.join(dataset_path, 'test')

print(f"Updated train dataloader paths:")
print(f"  data_root: {config.train_dataloader.dataset.data_root}")
print(f"  ann_file: {config.train_dataloader.dataset.ann_file}")
print(f"  data_prefix.img: {config.train_dataloader.dataset.data_prefix.img}")
print(f"Updated val dataloader paths:")
print(f"  data_root: {config.val_dataloader.dataset.data_root}")
print(f"  ann_file: {config.val_dataloader.dataset.ann_file}")
print(f"  data_prefix.img: {config.val_dataloader.dataset.data_prefix.img}")
print(f"Updated test dataloader paths:")
print(f"  data_root: {config.test_dataloader.dataset.data_root}")
print(f"  ann_file: {config.test_dataloader.dataset.ann_file}")
print(f"  data_prefix.img: {config.test_dataloader.dataset.data_prefix.img}")

# Update evaluators
config.val_evaluator.ann_file = os.path.join(dataset_path, 'val', 'annotations.json')
config.test_evaluator.ann_file = os.path.join(dataset_path, 'test', 'annotations.json')

# Update work directory
config.work_dir = work_dir

# Remove GPU settings to ensure flexibility for single or multiple GPUs
if hasattr(config, 'gpu_ids'):
    del config.gpu_ids

# Configure distributed training
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
for pipeline in [config.train_dataloader.dataset.pipeline, config.val_dataloader.dataset.pipeline, config.test_dataloader.dataset.pipeline]:
    for step in pipeline:
        if step['type'] == 'Normalize':
            step.update(norm_values_pipeline)

# Update normalization values in the config directly
config.normalization_values = normalization_values

# Ensure the work directory exists
os.makedirs(work_dir, exist_ok=True)

# Print the final configuration for debugging
print(config.pretty_text)

# Save the updated config
config.dump(f'{work_dir}/updated_config.py')

# Debug dataset loading
print("\nDebugging dataset loading:")
debug_dataset_loader(config, 'train')
debug_dataset_loader(config, 'val')
debug_dataset_loader(config, 'test')
