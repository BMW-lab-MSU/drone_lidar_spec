import sys
from mmengine.config import Config
import os

# Get arguments from the command line
config_path = sys.argv[1]
test_dataset_path = sys.argv[2]
work_dir = sys.argv[3]
gpu_id = int(sys.argv[4])  # Get the GPU ID from the arguments

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

# Ensure the work directory exists
os.makedirs(work_dir, exist_ok=True)

# Print the final configuration for debugging
print(config.pretty_text)

# Save the updated config
config.dump(f'{work_dir}/updated_config.py')
