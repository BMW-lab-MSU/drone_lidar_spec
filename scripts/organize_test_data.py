import os
import shutil
import json
import sys

"""
Splits files by tilt angle within each specified directory and creates separate annotations.json files for each angle.

Args:
    base_path (str): The base path containing the directories to be processed. Each directory should contain images and an annotations.json file.

Example Directory Structure:
- base_path/
    - boost_all/
    - boost_full/
    - boost_partial/
    - boost_prop-only/
    - noboost_all/
    - noboost_full/
    - noboost_partial/
    - noboost_prop-only/
        - annotations.json
        - image files (e.g., stan-fpv-17-05-56-tilt-0-fr-10_range_bin=121_time_slice=11.png)

Output:
    Each directory will be split into subdirectories by tilt angle (0, 10, ..., 90), each containing the corresponding images and a new annotations.json file.

Usage:
    python organize_test_data.py <base_path>

Example:
    python organize_test_data.py /path/to/testing_data
"""

def split_files_by_angle(base_path):
    for root, dirs, files in os.walk(base_path):
        if 'annotations.json' in files:
            annotations_file = os.path.join(root, 'annotations.json')
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            angle_bins = {str(i): [] for i in range(0, 91, 10)}

            for file in files:
                if file.endswith('.png'):
                    angle = file.split('tilt-')[1].split('-')[0]
                    angle_dir = os.path.join(root, f'tilt_{angle}')
                    if not os.path.exists(angle_dir):
                        os.makedirs(angle_dir)
                    angle_bins[angle].append(file)
                    shutil.move(os.path.join(root, file), os.path.join(angle_dir, file))

            for angle, files in angle_bins.items():
                angle_dir = os.path.join(root, f'tilt_{angle}')
                new_annotations = {
                    'images': [],
                    'annotations': []
                }

                for image in annotations['images']:
                    if any(file in image['file_name'] for file in files):
                        new_annotations['images'].append(image)

                for annotation in annotations['annotations']:
                    if any(image['id'] == annotation['image_id'] for image in new_annotations['images']):
                        new_annotations['annotations'].append(annotation)

                with open(os.path.join(angle_dir, 'annotations.json'), 'w') as f:
                    json.dump(new_annotations, f, indent=4)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python split_files_by_angle.py <base_path>")
        sys.exit(1)
    
    base_path = sys.argv[1]
    split_files_by_angle(base_path)
