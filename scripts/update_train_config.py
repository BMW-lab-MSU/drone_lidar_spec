import sys
from mmengine.config import Config
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to print contents of a directory
def print_directory_contents(path):
    logger.info(f"Contents of {path}:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            logger.info(f"{subindent}{f}")

# Function to print annotations content
def print_annotations_content(path):
    with open(path, 'r') as f:
        annotations = json.load(f)
    logger.info(f"Contents of {path}:")
    logger.info(json.dumps(annotations, indent=2)[:500])  # Print first 500 characters to avoid huge output

    # Check required sections
    required_sections = ['images', 'annotations', 'categories']
    for section in required_sections:
        if section not in annotations:
            logger.error(f"Missing section: {section}")

    # Log categories and bounding boxes
    for category in annotations.get('categories', []):
        logger.info(f"Category ID: {category['id']}, Name: {category['name']}")
    for annotation in annotations.get('annotations', []):
        logger.info(f"Annotation ID: {annotation['id']}, Image ID: {annotation['image_id']}, "
                    f"Category ID: {annotation['category_id']}, Bounding Box: {annotation['bbox']}, Area: {annotation['area']}")

    # Print the number of images and annotations
    logger.info(f"Number of images: {len(annotations['images'])}")
    logger.info(f"Number of annotations: {len(annotations['annotations'])}")

# Function to ensure path ends with a slash
def ensure_trailing_slash(path):
    return path if path.endswith('/') else path + '/'

# Get arguments from the command line
config_path = sys.argv[1]
dataset_path = ensure_trailing_slash(sys.argv[2])
work_dir = ensure_trailing_slash(sys.argv[3])
normalization_method = sys.argv[4]  # Get the normalization method

# Load the config file
config = Config.fromfile(config_path)

# Print initial configuration for debugging
logger.info("Initial train_dataloader.dataset configuration:")
logger.info(config.train_dataloader.dataset)
logger.info("Initial val_dataloader.dataset configuration:")
logger.info(config.val_dataloader.dataset)
logger.info("Initial test_dataloader.dataset configuration:")
logger.info(config.test_dataloader.dataset)

# Access nested dataset for train_dataloader if it's a RepeatDataset
train_dataset = config.train_dataloader.dataset
if train_dataset.type == 'RepeatDataset':
    train_dataset = train_dataset.dataset

# Print directory contents
print_directory_contents(os.path.join(dataset_path, 'train'))
print_directory_contents(os.path.join(dataset_path, 'val'))
print_directory_contents(os.path.join(dataset_path, 'test'))

# Print annotations contents
print_annotations_content(os.path.join(dataset_path, 'train', 'annotations.json'))
print_annotations_content(os.path.join(dataset_path, 'val', 'annotations.json'))
print_annotations_content(os.path.join(dataset_path, 'test', 'annotations.json'))

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
    logger.info("FOUND TRAIN DATA PREFIX")
    train_dataset.data_prefix['img'] = os.path.join(dataset_path, 'train') + '/'
else:
    logger.info("TRAIN DATA PREFIX NOT FOUND")

if 'data_prefix' in config.val_dataloader.dataset:
    logger.info("FOUND VAL DATA PREFIX")
    config.val_dataloader.dataset.data_prefix['img'] = os.path.join(dataset_path, 'val') + '/'
else:
    logger.info("VAL DATA PREFIX NOT FOUND")

if 'data_prefix' in config.test_dataloader.dataset:
    logger.info("FOUND TEST DATA PREFIX")
    config.test_dataloader.dataset.data_prefix['img'] = os.path.join(dataset_path, 'test') + '/'
else:
    logger.info("TEST DATA PREFIX NOT FOUND")

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
logger.info("Updated train_dataloader.dataset configuration:")
logger.info(config.train_dataloader.dataset)
logger.info("Updated val_dataloader.dataset configuration:")
logger.info(config.val_dataloader.dataset)
logger.info("Updated test_dataloader.dataset configuration:")
logger.info(config.test_dataloader.dataset)

logger.info(config.pretty_text)

# Save the updated config
config.dump(os.path.join(work_dir, 'updated_config.py'))
