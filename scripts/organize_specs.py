import os
import json
import shutil
import argparse
import logging

"""
Organize Spectrograms Script

This script organizes spectrograms into datasets based on the `Fill Factor` value in the `details.txt` file and splits them into train/val/test sets.
It creates four datasets: one for each `Fill Factor` value ("b'full'", "b'prop only'", "b'partial'") and one that includes all fill factors.
Each dataset is split into 22 images for training, 5 for validation, and 5 for testing from each folder containing 32 images.

Usage:
    python organize_spectrograms.py --input_dir all_spectrograms_chunked --output_dir organized_spectrograms
"""

# Fill Factor categories
fill_factors = ["b'full'", "b'prop only'", "b'partial'"]

def load_details(file_path):
    with open(file_path, 'r') as f:
        details = {}
        for line in f:
            key, value = line.strip().split(': ')
            details[key.strip()] = value.strip()
        return details

def collect_files(input_dir):
    all_files = []
    for root, _, files in os.walk(input_dir):
        if 'details.txt' in files:
            details = load_details(os.path.join(root, 'details.txt'))
            if 'stan-fpv' in details.get('drone_name', ''):
                fill_factor = details.get('Fill Factor')
                all_files.append({
                    'root': root,
                    'details': details,
                    'fill_factor': fill_factor,
                    'drone_name': details['drone_name']
                })
    return all_files

def copy_files(files, destination_dir, start_index, end_index):
    moved_images_count = 0
    for file_info in files[start_index:end_index]:
        src_folder = os.path.join(file_info['root'], 'Raw')
        for file_name in os.listdir(src_folder):
            shutil.copy(os.path.join(src_folder, file_name), os.path.join(destination_dir, file_name))
            moved_images_count += 1
            if moved_images_count % 1000 == 0:
                logging.info(f"Moved {moved_images_count} images so far.")

def merge_annotations(all_files, destination_dir):
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

    for file_info in all_files:
        with open(os.path.join(file_info['root'], 'annotations.json'), 'r') as f:
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
    for factor in fill_factors + ['all']:
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, factor, split), exist_ok=True)

    for fill_factor in fill_factors:
        factor_files = [f for f in all_files if f['fill_factor'] == fill_factor]
        logging.info(f"Processing Fill Factor: {fill_factor}, {len(factor_files)} directories found.")

        for file_info in factor_files:
            train_dir = os.path.join(output_dir, fill_factor, 'train')
            val_dir = os.path.join(output_dir, fill_factor, 'val')
            test_dir = os.path.join(output_dir, fill_factor, 'test')

            copy_files([file_info], train_dir, 0, 22)
            copy_files([file_info], val_dir, 22, 27)
            copy_files([file_info], test_dir, 27, 32)
            logging.info(f"Copied files from {file_info['root']} to {fill_factor} dataset.")

    # Create the 'all' dataset
    for split in ['train', 'val', 'test']:
        all_files_split = []
        for fill_factor in fill_factors:
            split_files = [f for f in all_files if f['fill_factor'] == fill_factor]
            for file_info in split_files:
                if split == 'train':
                    all_files_split.extend([file_info] * 22)
                elif split == 'val':
                    all_files_split.extend([file_info] * 5)
                elif split == 'test':
                    all_files_split.extend([file_info] * 5)

        all_files_split = all_files_split[:32]  # Ensure each split has exactly 32 images
        split_dir = os.path.join(output_dir, 'all', split)
        copy_files(all_files_split, split_dir, 0, len(all_files_split))
        logging.info(f"Created 'all' dataset split: {split}")

    for fill_factor in fill_factors + ['all']:
        factor_files = [f for f in all_files if f['fill_factor'] == fill_factor or fill_factor == 'all']
        merge_annotations(factor_files, os.path.join(output_dir, fill_factor))
        logging.info(f"Merged annotations for Fill Factor: {fill_factor}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize spectrograms into datasets and split into train/val/test sets.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input directory containing spectrograms.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory where organized datasets will be saved.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
