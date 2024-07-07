import os
import json
import shutil
import argparse
import logging

"""
Organize Spectrograms Script

This script organizes spectrograms into datasets based on the `Fill Factor` value in the `details.txt` file and splits them into train/val/test sets.
It creates four datasets: one for each `Fill Factor` value ("full", "prop-only", "partial") and one that includes all fill factors.
Each dataset splits the files from each subdirectory such that 22 images go to training, 5 to validation, and 5 to testing.

Usage:
    python organize_spectrograms.py --input_dir all_spectrograms_chunked --output_dir organized_spectrograms
"""

# Fill Factor categories
fill_factors = ["full", "prop-only", "partial"]

def load_details(file_path):
    with open(file_path, 'r') as f:
        details = {}
        for line in f:
            if ': ' in line:
                key, value = line.strip().split(': ', 1)
                details[key.strip()] = value.strip()
        return details

def collect_files(input_dir):
    all_files = []
    for root, _, files in os.walk(input_dir):
        if 'details.txt' in files:
            details = load_details(os.path.join(root, 'details.txt'))
            if 'stan-fpv' in details.get('drone_name', ''):
                fill_factor = details.get('Fill Factor')
                if fill_factor:
                    fill_factor = fill_factor.strip("b'").rstrip("'").replace(" ", "-")
                    all_files.append({
                        'root': root,
                        'details': details,
                        'fill_factor': fill_factor,
                        'drone_name': details['drone_name']
                    })
    return all_files

def copy_files(src_folder, destination_dir, start_index, end_index):
    moved_images_count = 0
    with os.scandir(src_folder) as entries:
        file_list = [entry for entry in entries if entry.is_file() and entry.name.endswith('.png')]
        for entry in file_list[start_index:end_index]:
            shutil.copy(entry.path, os.path.join(destination_dir, entry.name))
            moved_images_count += 1
            if moved_images_count % 1000 == 0:
                logging.info(f"Moved {moved_images_count} images so far from {src_folder}.")
    logging.info(f"Moved {len(file_list[start_index:end_index])} files from {src_folder}")

def merge_annotations(files, destination_dir):
    combined_annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "drone_frequency",
                "supercategory": "object"
            }
        ]
    }
    image_id_offset = 0
    annotation_id_offset = 0

    for file_info in files:
        annotations_path = os.path.join(os.path.dirname(file_info['root']), 'annotations.json')
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
                for image in annotations['images']:
                    image['id'] += image_id_offset
                for annotation in annotations['annotations']:
                    annotation['id'] += annotation_id_offset
                    annotation['image_id'] += image_id_offset

                combined_annotations['images'].extend(annotations['images'])
                combined_annotations['annotations'].extend(annotations['annotations'])
                image_id_offset += len(annotations['images'])
                annotation_id_offset += len(annotations['annotations'])

    with open(os.path.join(destination_dir, 'annotations.json'), 'w') as f:
        json.dump(combined_annotations, f, indent=4)

def main(input_dir, output_dir):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    all_files = collect_files(input_dir)
    logging.info(f"Collected {len(all_files)} valid directories with 'stan-fpv' drone name.")

    # Ensure output directories exist
    for factor in fill_factors:
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, factor, split), exist_ok=True)

    # Process each fill factor
    for fill_factor in fill_factors:
        factor_files = [f for f in all_files if f['fill_factor'] == fill_factor]
        logging.info(f"Processing Fill Factor: {fill_factor}, {len(factor_files)} directories found.")

        for file_info in factor_files:
            src_folder = os.path.join(file_info['root'], 'Raw')
            if os.path.exists(src_folder):
                # Copy files for train
                train_dir = os.path.join(output_dir, fill_factor, 'train')
                copy_files(src_folder, train_dir, 0, 22)

                # Copy files for val
                val_dir = os.path.join(output_dir, fill_factor, 'val')
                copy_files(src_folder, val_dir, 22, 27)

                # Copy files for test
                test_dir = os.path.join(output_dir, fill_factor, 'test')
                copy_files(src_folder, test_dir, 27, 32)

        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(output_dir, fill_factor, split)
            merge_annotations(factor_files, split_dir)
            logging.info(f"Copied and merged annotations for {split} split of Fill Factor: {fill_factor}")

    # Create the 'all' dataset by combining splits from the other three
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, 'all', split)
        os.makedirs(split_dir, exist_ok=True)
        for fill_factor in fill_factors:
            factor_split_dir = os.path.join(output_dir, fill_factor, split)
            for file_name in os.listdir(factor_split_dir):
                src_path = os.path.join(factor_split_dir, file_name)
                dst_path = os.path.join(split_dir, file_name)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)
        logging.info(f"Created 'all' dataset split: {split}")

    # Merge annotations for 'all' dataset
    for split in ['train', 'val', 'test']:
        split_files = []
        for fill_factor in fill_factors:
            factor_split_dir = os.path.join(output_dir, fill_factor, split)
            for root, _, files in os.walk(factor_split_dir):
                if 'details.txt' in files:
                    split_files.append({
                        'root': root,
                        'details': load_details(os.path.join(root, 'details.txt')),
                        'fill_factor': fill_factor,
                        'drone_name': 'stan-fpv'
                    })
        split_dir = os.path.join(output_dir, 'all', split)
        merge_annotations(split_files, split_dir)
        logging.info(f"Merged annotations for 'all' dataset split: {split}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize spectrograms into datasets and split into train/val/test sets.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input directory containing spectrograms.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory where organized datasets will be saved.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
