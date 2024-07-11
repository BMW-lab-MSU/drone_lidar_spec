import os
import shutil
import json
import sys

"""
Splits files by tilt angle within each specified directory and creates separate annotations.json files for each angle.

Args:
    input_base_path (str): The input base path containing the directories to be processed. Each directory should contain images and an annotations.json file.
    output_base_path (str): The output base path where the new directories will be created and organized.

Example Input Directory Structure:
- input_base_path/
    ├── boost_all/
    │   ├── test/
    │   │   ├── annotations.json
    │   │   ├── image files (e.g., stan-fpv-17-05-56-tilt-0-fr-10_range_bin=121_time_slice=11.png)
    │   ├── train/
    │   └── val/
    ├── noboost_partial_SF/
        ├── test/
        │   ├── annotations.json
        │   ├── image files
        ├── train/
        └── val/

Example Output Directory Structure:
- output_base_path/
    ├── boost_all/
    │   ├── tilt_0/
    │   │   ├── annotations.json
    │   │   ├── image files
    │   ├── tilt_10/
    │   │   ├── annotations.json
    │   │   ├── image files
    │   └── ... (other tilt angle folders)
    └── noboost_partial_SF/
        ├── tilt_0/
        │   ├── annotations.json
        │   ├── image files
        ├── tilt_10/
        │   ├── annotations.json
        │   ├── image files
        └── ... (other tilt angle folders)

Output:
    Each directory will be split into subdirectories by tilt angle (0, 10, ..., 90), each containing the corresponding images and a new annotations.json file. If a directory contains only one tilt angle, images and annotations will be placed directly in the parent directory.

Usage:
    python organize_test_data.py <input_base_path> <output_base_path>

Example:
    python organize_test_data.py /path/to/input_data /path/to/output_data
"""

def split_files_by_angle(input_base_path, output_base_path):
    for root, dirs, files in os.walk(input_base_path):
        if 'test' in dirs:
            print(f"Processing directory: {root}")
            test_dir = os.path.join(root, 'test')
            relative_path = os.path.relpath(root, input_base_path)
            new_root_dir = os.path.join(output_base_path, relative_path)

            # Create the root directory in the output path
            os.makedirs(new_root_dir, exist_ok=True)

            annotations_file = os.path.join(test_dir, 'annotations.json')
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)

            angle_bins = {str(i): [] for i in range(0, 91, 10)}

            for file in os.listdir(test_dir):
                if file.endswith('.png'):
                    angle = file.split('tilt-')[1].split('-')[0]
                    angle_bins[angle].append(file)

            # Determine if there is only one tilt angle
            active_angles = [angle for angle, files in angle_bins.items() if files]
            single_angle = len(active_angles) == 1

            for angle, files in angle_bins.items():
                if files:  # Only process angles that have files
                    if single_angle:
                        angle_dir = new_root_dir
                    else:
                        angle_dir = os.path.join(new_root_dir, f'tilt_{angle}')
                        os.makedirs(angle_dir, exist_ok=True)
                    
                    for file in files:
                        shutil.copy(os.path.join(test_dir, file), os.path.join(angle_dir, file))

                    new_annotations = {
                        'images': [],
                        'annotations': [],
                        'categories': annotations.get('categories', [
                            {
                                "id": 1,
                                "name": "drone_frequency",
                                "supercategory": "object"
                            }
                        ])
                    }

                    for image in annotations['images']:
                        if any(file in image['file_name'] for file in files):
                            new_annotations['images'].append(image)

                    for annotation in annotations['annotations']:
                        if any(image['id'] == annotation['image_id'] for image in new_annotations['images']):
                            new_annotations['annotations'].append(annotation)

                    with open(os.path.join(angle_dir, 'annotations.json'), 'w') as f:
                        json.dump(new_annotations, f, indent=4)
                    
                    print(f"Created {len(files)} files and annotations in {angle_dir}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python split_files_by_angle.py <input_base_path> <output_base_path>")
        sys.exit(1)
    
    input_base_path = sys.argv[1]
    output_base_path = sys.argv[2]
    split_files_by_angle(input_base_path, output_base_path)
