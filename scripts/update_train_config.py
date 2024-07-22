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

# Ensure the data object exists
if not hasattr(config, 'data'):
    raise AttributeError("The configuration file does not contain a 'data' attribute.")

# Update dataset paths in the config
config.data.train.data_root = dataset_path
config.data.val.data_root = dataset_path
config.data.test.data_root = dataset_path
config.data.train.ann_file = f'{dataset_path}/train/annotations.json'
config.data.val.ann_file = f'{dataset_path}/val/annotations.json'
config.data.test.ann_file = f'{dataset_path}/test/annotations.json'
config.data.train.img_prefix = f'{dataset_path}/train/'
config.data.val.img_prefix = f'{dataset_path}/val/'
config.data.test.img_prefix = f'{dataset_path}/test/'
config.val_evaluator.ann_file = f'{dataset_path}/val/annotations.json'
config.test_evaluator.ann_file = f'{dataset_path}/test/annotations.json'

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
for phase in ['train', 'val', 'test']:
    if hasattr(config.data, phase) and hasattr(config.data[phase], 'pipeline'):
        pipeline = config.data[phase].pipeline
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
