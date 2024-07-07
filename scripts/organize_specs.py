import os
import json
import shutil
import argparse
import logging
import re
from collections import defaultdict

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
                    chunk_number = int(re.search(r'split_(\d+)', root).group(1))
                    all_files.append({
                        'root': root,
                        'details': details,
                        'fill_factor': fill_factor,
                        'drone_name': details['drone_name'],
                        'chunk_number': chunk_number
                    })
    return all_files

def copy_files(src_folder, destination_dir, start_index, end_index):
    moved_images_count = 0
    image_files = []
    with os.scandir(src_folder) as entries:
        file_list = [entry for entry in entries if entry.is_file() and entry.name.endswith('.png')]
        for entry in file_list[start_index:end_index]:
            shutil.copy(entry.path, os.path.join(destination_dir, entry.name))
            image_files.append(entry.name)
            moved_images_count += 1
            if moved_images_count % 1000 == 0:
                logging.info(f"Moved {moved_images_count} images so far from {src_folder}.")
    logging.info(f"Moved {len(file_list[start_index:end_index])} files from {src_folder}")
    return image_files

def merge_annotations(files, master_annotations_path):
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

    for file_info in files:
        annotations_path = os.path.join(os.path.dirname(file_info['root']), 'annotations.json')
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
                combined_annotations['images'].extend(annotations['images'])
                combined_annotations['annotations'].extend(annotations['annotations'])

    os.makedirs(os.path.dirname(master_annotations_path), exist_ok=True)
    with open(master_annotations_path, 'w') as f:
        json.dump(combined_annotations, f, indent=4)

def create_lookup_dicts(master_annotations):
    image_lookup = {}
    annotation_lookup = defaultdict(list)
    for image in master_annotations["images"]:
        image_lookup[image["file_name"]] = image
    for annotation in master_annotations["annotations"]:
        annotation_lookup[annotation["image_id"]].append(annotation)
    return image_lookup, annotation_lookup

def filter_annotations(image_files_set, image_lookup, annotation_lookup):
    logging.info("Starting to filter annotations.")
    filtered_annotations = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for file_name in image_files_set:
        if file_name in image_lookup:
            image = image_lookup[file_name]
            filtered_annotations["images"].append(image)
            image_id = image["id"]
            filtered_annotations["annotations"].extend(annotation_lookup[image_id])

    logging.info(f"Filtered {len(filtered_annotations['images'])} images and {len(filtered_annotations['annotations'])} annotations.")
    return filtered_annotations

def main(input_dir, output_dir):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    all_files = collect_files(input_dir)
    logging.info(f"Collected {len(all_files)} valid directories with 'stan-fpv' drone name.")

    master_annotations_path = os.path.join(output_dir, 'master_annotations.json')
    merge_annotations(all_files, master_annotations_path)
    logging.info(f"Master annotations file created at {master_annotations_path}")

    # Ensure output directories exist
    for factor in fill_factors:
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, factor, split), exist_ok=True)

    # Move all images first and track moved images
    split_image_files = defaultdict(lambda: defaultdict(list))
    for fill_factor in fill_factors:
        factor_files = [f for f in all_files if f['fill_factor'] == fill_factor]
        logging.info(f"Processing Fill Factor: {fill_factor}, {len(factor_files)} directories found.")

        for file_info in factor_files:
            src_folder = os.path.join(file_info['root'], 'Raw')
            if os.path.exists(src_folder):
                # Copy files for train
                train_dir = os.path.join(output_dir, fill_factor, 'train')
                split_image_files[fill_factor]['train'].extend(copy_files(src_folder, train_dir, 0, 22))

                # Copy files for val
                val_dir = os.path.join(output_dir, fill_factor, 'val')
                split_image_files[fill_factor]['val'].extend(copy_files(src_folder, val_dir, 22, 27))

                # Copy files for test
                test_dir = os.path.join(output_dir, fill_factor, 'test')
                split_image_files[fill_factor]['test'].extend(copy_files(src_folder, test_dir, 27, 32))

    logging.info("All images moved to their respective folders.")

    # Load master annotations once and create lookup dictionaries
    with open(master_annotations_path, 'r') as f:
        master_annotations = json.load(f)
    logging.info("Master annotations loaded.")
    
    image_lookup, annotation_lookup = create_lookup_dicts(master_annotations)
    logging.info("Lookup dictionaries created.")

    # Filter and copy annotations for each split
    for split in ['train', 'val', 'test']:
        for fill_factor in fill_factors:
            split_dir = os.path.join(output_dir, fill_factor, split)
            image_files_set = set(split_image_files[fill_factor][split])
            logging.info(f"Filtering annotations for {split} split of {fill_factor} fill factor.")
            filtered_annotations = filter_annotations(image_files_set, image_lookup, annotation_lookup)
            filtered_annotations["categories"] = master_annotations["categories"]
            with open(os.path.join(split_dir, 'annotations.json'), 'w') as f:
                json.dump(filtered_annotations, f, indent=4)
            logging.info(f"Filtered annotations for {split} split of {fill_factor} fill factor saved.")

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

    # Filter annotations for 'all' dataset
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, 'all', split)
        image_files = [f for f in os.listdir(split_dir) if f.endswith('.png')]
        image_files_set = set(image_files)
        logging.info(f"Filtering annotations for 'all' dataset split: {split}")
        filtered_annotations = filter_annotations(image_files_set, image_lookup, annotation_lookup)
        filtered_annotations["categories"] = master_annotations["categories"]
        with open(os.path.join(split_dir, 'annotations.json'), 'w') as f:
            json.dump(filtered_annotations, f, indent=4)
        logging.info(f"Filtered annotations for 'all' dataset split: {split} saved.")

    # Clean up the master annotations file
    os.remove(master_annotations_path)
    logging.info(f"Deleted the master annotations file: {master_annotations_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize spectrograms into datasets and split into train/val/test sets.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input directory containing spectrograms.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory where organized datasets will be saved.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
