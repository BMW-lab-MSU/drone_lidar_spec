import sys
import os
import json
import re
from mmengine.config import Config

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

def update_config(config_path, dataset_path, work_dir, normalization_method):
    try:
        # Load the config file
        config = Config.fromfile(config_path)
        print("Original configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Check annotations for train, val, and test datasets
    for phase in ['train', 'val', 'test']:
        if not check_annotations(dataset_path, phase):
            print("Error: One or more datasets are invalid. Exiting...")
            return

    # Update dataset paths in the config
    for phase in ['train', 'val', 'test']:
        if hasattr(config.data, phase):
            print(f"Updating {phase} dataset paths")
            phase_data = getattr(config.data, phase)
            phase_data.data_root = dataset_path
            phase_data.ann_file = os.path.join(dataset_path, phase, 'annotations.json')
            phase_data.img_prefix = os.path.join(dataset_path, phase)
    
    # Update the evaluators
    if hasattr(config, 'val_evaluator'):
        print("Updating val evaluator")
        config.val_evaluator.ann_file = os.path.join(dataset_path, 'val', 'annotations.json')
    
    if hasattr(config, 'test_evaluator'):
        print("Updating test evaluator")
        config.test_evaluator.ann_file = os.path.join(dataset_path, 'test', 'annotations.json')

    # Update the work directory
    print("Updating work directory")
    config.work_dir = work_dir

    # Ensure distributed training is properly configured
    print("Updating distributed training configuration")
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
    else:
        # Default values if normalization_method is not recognized
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]

    # Regular expressions to find and replace the mean and std values
    mean_pattern = re.compile(r'mean=\[.*?\]')
    std_pattern = re.compile(r'std=\[.*?\]')

    new_mean = f"mean={mean}"
    new_std = f"std={std}"

    config_text = config.pretty_text
    config_text = mean_pattern.sub(new_mean, config_text)
    config_text = std_pattern.sub(new_std, config_text)

    # Write the updated config back to the file
    updated_config_path = os.path.join(work_dir, 'updated_config.py')
    with open(updated_config_path, 'w') as file:
        file.write(config_text)

    print(f"\nConfiguration update successful. Config file saved at: {updated_config_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python update_config.py <config_path> <dataset_path> <work_dir> <normalization_method>")
    else:
        update_config(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
