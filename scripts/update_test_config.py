import sys
from mmengine.config import Config
import os

# Get arguments from the command line
config_path = sys.argv[1]
test_dataset_path = sys.argv[2]
work_dir = sys.argv[3]
gpu_id = int(sys.argv[4])  # Get the GPU ID from the arguments
normalization_method = sys.argv[5]  # Get the normalization method

# Load the config file
config = Config.fromfile(config_path)

# Update testing dataset paths in the config
config.test_dataloader.dataset.data_root = test_dataset_path
config.test_dataloader.dataset.ann_file = f'{test_dataset_path}/annotations.json'
config.test_dataloader.dataset.data_prefix.img = test_dataset_path
config.test_evaluator.ann_file = f'{test_dataset_path}/annotations.json'

# Update the work directory
config.work_dir = work_dir

# Update the GPU ID
config.gpu_ids = [gpu_id]

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
    std = [1.0/255.0, 1.0/255.0, 1.0/255.0]

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
