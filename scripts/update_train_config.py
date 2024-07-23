import sys
from mmengine.config import Config
import os
import json

def check_annotations(dataset_path, phase):
    annotations_file = f'{dataset_path}/{phase}/annotations.json'
    if not os.path.exists(annotations_file):
        print(f"{phase} annotations file not found at {annotations_file}")
        return False

    with open(annotations_file, 'r') as f:
        data = json.load(f)

    num_images = len(data.get('images', []))
    num_annotations = len(data.get('annotations', []))

    if num_images == 0 or num_annotations == 0:
        print(f"{phase} dataset is empty. Found {num_images} images and {num_annotations} annotations.")
        return False
    else:
        print(f"{phase} dataset contains {num_images} images and {num_annotations} annotations.")
        return True

def update_config(config_path, dataset_path, work_dir, normalization_method):
    try:
        # Load the config file
        config = Config.fromfile(config_path)
        print("Original configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    try:
        # Print the original configuration for debugging
        print("Original configuration:\n")
        print(config.pretty_text)
    except Exception as e:
        print(f"Error printing original configuration: {e}")
        return

    try:
        # Ensure the data object exists
        if not hasattr(config, 'data'):
            raise AttributeError("The configuration file does not contain a 'data' attribute.")

        # Check annotations for train, val, and test datasets
        valid_train = check_annotations(dataset_path, 'train')
        valid_val = check_annotations(dataset_path, 'val')
        valid_test = check_annotations(dataset_path, 'test')

        if not (valid_train and valid_val and valid_test):
            print("One or more datasets are invalid. Exiting...")
            return

        # Update dataset paths in the config
        if hasattr(config.data, 'train'):
            print("Updating train dataset paths")
            config.data.train.data_root = dataset_path
            config.data.train.ann_file = f'{dataset_path}/train/annotations.json'
            config.data.train.img_prefix = f'{dataset_path}/train/'
        else:
            print("Train dataset not found in config")

        if hasattr(config.data, 'val'):
            print("Updating val dataset paths")
            config.data.val.data_root = dataset_path
            config.data.val.ann_file = f'{dataset_path}/val/annotations.json'
            config.data.val.img_prefix = f'{dataset_path}/val/'
        else:
            print("Val dataset not found in config")

        if hasattr(config.data, 'test'):
            print("Updating test dataset paths")
            config.data.test.data_root = dataset_path
            config.data.test.ann_file = f'{dataset_path}/test/annotations.json'
            config.data.test.img_prefix = f'{dataset_path}/test/'
        else:
            print("Test dataset not found in config")

        # Update the evaluators
        if hasattr(config, 'val_evaluator'):
            print("Updating val evaluator")
            config.val_evaluator.ann_file = f'{dataset_path}/val/annotations.json'
        else:
            print("Val evaluator not found in config")

        if hasattr(config, 'test_evaluator'):
            print("Updating test evaluator")
            config.test_evaluator.ann_file = f'{dataset_path}/test/annotations.json'
        else:
            print("Test evaluator not found in config")
    except Exception as e:
        print(f"Error updating dataset paths or evaluators: {e}")
        return

    try:
        # Update the work directory
        print("Updating work directory")
        config.work_dir = work_dir
    except Exception as e:
        print(f"Error updating work directory: {e}")
        return

    try:
        # Ensure distributed training is properly configured
        print("Updating distributed training configuration")
        config.dist_params = dict(backend='nccl')
        config.launcher = 'slurm'
    except Exception as e:
        print(f"Error updating distributed training configuration: {e}")
        return

    try:
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

        # Update normalization values
        normalization_values = {
            'mean': mean,
            'std': std
        }

        # Update normalization in the data preprocessor
        if hasattr(config.model, 'data_preprocessor'):
            print("Updating data preprocessor normalization values")
            config.model.data_preprocessor.mean = mean
            config.model.data_preprocessor.std = std
        else:
            print("Data preprocessor not found in config")
    except Exception as e:
        print(f"Error updating normalization values: {e}")
        return

    try:
        # Update normalization in the training, validation, and test pipelines
        norm_values_pipeline = dict(type='Normalize', mean=mean, std=std, to_rgb=True)
        for phase in ['train', 'val', 'test']:
            if hasattr(config.data, phase) and hasattr(config.data[phase], 'pipeline'):
                print(f"Updating {phase} pipeline normalization values")
                pipeline = config.data[phase].pipeline
                for step in pipeline:
                    if step['type'] == 'Normalize':
                        step.update(norm_values_pipeline)
            else:
                print(f"{phase.capitalize()} pipeline not found in config")
    except Exception as e:
        print(f"Error updating normalization in pipelines: {e}")
        return

    try:
        # Update normalization values in the config directly
        print("Updating normalization values in the config")
        config.normalization_values = normalization_values
    except Exception as e:
        print(f"Error updating normalization values in config: {e}")
        return

    try:
        # Ensure the work directory exists
        os.makedirs(work_dir, exist_ok=True)
    except Exception as e:
        print(f"Error ensuring work directory exists: {e}")
        return

    try:
        # Update dataloaders
        if hasattr(config, 'train_dataloader'):
            print("Updating train dataloader")
            config.train_dataloader.dataset.data_root = dataset_path
            config.train_dataloader.dataset.ann_file = f'{dataset_path}/train/annotations.json'
            config.train_dataloader.dataset.data_prefix = dict(img=f'{dataset_path}/train/')
        else:
            print("Train dataloader not found in config")

        if hasattr(config, 'val_dataloader'):
            print("Updating val dataloader")
            config.val_dataloader.dataset.data_root = dataset_path
            config.val_dataloader.dataset.ann_file = f'{dataset_path}/val/annotations.json'
            config.val_dataloader.dataset.data_prefix = dict(img=f'{dataset_path}/val/')
        else:
            print("Val dataloader not found in config")

        if hasattr(config, 'test_dataloader'):
            print("Updating test dataloader")
            config.test_dataloader.dataset.data_root = dataset_path
            config.test_dataloader.dataset.ann_file = f'{dataset_path}/test/annotations.json'
            config.test_dataloader.dataset.data_prefix = dict(img=f'{dataset_path}/test/')
        else:
            print("Test dataloader not found in config")
    except Exception as e:
        print(f"Error updating dataloaders: {e}")
        return

    try:
        # Print the final configuration for debugging
        print("\nUpdated configuration:\n")
        print(config.pretty_text)
    except Exception as e:
        print(f"Error printing updated configuration: {e}")
        return

    try:
        # Save the updated config
        config.dump(f'{work_dir}/updated_config.py')
    except Exception as e:
        print(f"Error saving updated configuration: {e}")
        return

    print("\nConfiguration update successful. Config file saved at:", f'{work_dir}/updated_config.py')

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python update_config.py <config_path> <dataset_path> <work_dir> <normalization_method>")
    else:
        update_config(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
