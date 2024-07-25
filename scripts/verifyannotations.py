"""
This script verifies that the images and annotations match in the dataset folders.
It checks for missing images in the annotations and missing annotations for the images.

Usage:
    python verify_annotations.py <dataset_path>

Arguments:
    dataset_path: Path to the dataset directory. The directory should contain subfolders
                  'train', 'val', and 'test', each having 'images' and 'annotations' subfolders.
                  The 'annotations' subfolder should contain a file named 'annotations.json'.

Example:
    python verify_annotations.py /path/to/dataset

Output:
    The script will print the list of images missing in the annotations and the list of annotations
    missing corresponding images for each of the 'train', 'val', and 'test' folders.
    If all annotations and images match, it will indicate that as well.

Requirements:
    - Python 3.x
    - The dataset directory structure should be as follows:
        dataset/
        ├── train/
        │   ├── images/
        │   └── annotations/
        │       └── annotations.json
        ├── val/
        │   ├── images/
        │   └── annotations/
        │       └── annotations.json
        └── test/
            ├── images/
            └── annotations/
                └── annotations.json
"""

import os
import json
import argparse

def load_annotations(file_path):
    with open(file_path, 'r') as file:
        annotations = json.load(file)
    return annotations

def get_image_files(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def verify_annotations(images_folder, annotations_file):
    annotations = load_annotations(annotations_file)
    image_files = get_image_files(images_folder)

    annotated_images = {os.path.basename(ann['file_name']) for ann in annotations['images']}
    image_files_set = set(image_files)

    missing_in_annotations = image_files_set - annotated_images
    missing_in_images = annotated_images - image_files_set

    return missing_in_annotations, missing_in_images

def main(dataset_path):
    dataset_folders = ["train", "val", "test"]

    for folder in dataset_folders:
        annotations_file = os.path.join(dataset_path, folder, "annotations", "annotations.json")
        images_folder = os.path.join(dataset_path, folder, "images")

        if not os.path.exists(annotations_file):
            print(f"Annotations file not found: {annotations_file}")
            continue

        if not os.path.exists(images_folder):
            print(f"Images folder not found: {images_folder}")
            continue

        missing_in_annotations, missing_in_images = verify_annotations(images_folder, annotations_file)

        if missing_in_annotations:
            print(f"Images in {folder} folder missing in annotations:")
            for missing in missing_in_annotations:
                print(missing)

        if missing_in_images:
            print(f"Annotations in {folder} folder missing corresponding images:")
            for missing in missing_in_images:
                print(missing)

        if not missing_in_annotations and not missing_in_images:
            print(f"All annotations and images match in {folder} folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify that images and annotations match in dataset folders.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    args = parser.parse_args()
    main(args.dataset_path)
