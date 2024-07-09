"""
This script filters a dataset by a specified tilt value and creates a new dataset containing only the filtered images and annotations.

Usage:
    python filter_dataset.py <src_root> <dest_root> <tilt_value>

Arguments:
    src_root   : Path to the source root directory containing 'train', 'test', and 'val' folders.
    dest_root  : Path to the destination root directory to store the filtered dataset.
    tilt_value : Tilt value to filter images by (must be between 0 and 90, divisible by 10).

Example:
    python filter_dataset.py ./source_dataset ./filtered_dataset 90

The script will create a new directory structure under the destination root directory, containing only the images with the specified tilt value and updated annotations.

Input Directory Structure:
source_dataset/
├── train
│   ├── annotations.json
│   ├── image1-tilt-0.png
│   ├── image2-tilt-90.png
│   └── ...
├── test
│   ├── annotations.json
│   ├── image1-tilt-0.png
│   ├── image2-tilt-90.png
│   └── ...
└── val
    ├── annotations.json
    ├── image1-tilt-0.png
    ├── image2-tilt-90.png
    └── ...

Output Directory Structure:
filtered_dataset/
├── train
│   ├── annotations.json
│   ├── image2-tilt-90.png
│   └── ...
├── test
│   ├── annotations.json
│   ├── image2-tilt-90.png
│   └── ...
└── val
    ├── annotations.json
    ├── image2-tilt-90.png
    └── ...
"""

import os
import shutil
import json
import argparse

def create_filtered_dataset(src_root, dest_root, tilt_value):
    # Ensure destination root exists
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    # Process each subset directory (train, test, val)
    for subset in ['train', 'test', 'val']:
        src_dir = os.path.join(src_root, subset)
        dest_dir = os.path.join(dest_root, subset)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Filter images and copy to destination directory
        filtered_files = []
        for file_name in os.listdir(src_dir):
            if file_name.endswith('.png') and f"tilt-{tilt_value}" in file_name:
                shutil.copy(os.path.join(src_dir, file_name), dest_dir)
                filtered_files.append(file_name)

        # Filter annotations.json
        src_annotations = os.path.join(src_dir, 'annotations.json')
        dest_annotations = os.path.join(dest_dir, 'annotations.json')

        if os.path.exists(src_annotations):
            with open(src_annotations, 'r') as f:
                annotations = json.load(f)

            # Create a map from file name to image ID
            image_id_map = {image['file_name']: image['id'] for image in annotations['images'] if image['file_name'] in filtered_files}

            # Filter images to include only relevant images
            filtered_images = [image for image in annotations['images'] if image['file_name'] in filtered_files]

            # Filter annotations to include only relevant annotations
            filtered_annotations = [anno for anno in annotations['annotations'] if anno['image_id'] in image_id_map.values()]

            # Create new annotations structure
            new_annotations = {
                'images': filtered_images,
                'annotations': filtered_annotations,
                'categories': annotations['categories']
            }

            with open(dest_annotations, 'w') as f:
                json.dump(new_annotations, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Filter dataset by tilt value and create a new dataset.")
    parser.add_argument('src_root', type=str, help='Path to the source root directory')
    parser.add_argument('dest_root', type=str, help='Path to the destination root directory')
    parser.add_argument('tilt_value', type=int, choices=range(0, 91, 10), help='Tilt value to filter images by (must be between 0 and 90, divisible by 10)')
    
    args = parser.parse_args()

    create_filtered_dataset(args.src_root, args.dest_root, args.tilt_value)

if __name__ == "__main__":
    main()
