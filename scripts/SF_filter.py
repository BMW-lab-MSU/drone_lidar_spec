import os
import shutil
import json
import argparse

"""
Script to create a subset of files and annotations from a dataset based on a list of file name substrings.

This script copies a subset of images from the source directory to a new directory, 
and updates the annotations.json file to include only the annotations for the copied images.

Usage:
    python SF_filter.py --file_list /path/to/file_list.txt --input_dir /path/to/input_dir --output_dir /path/to/output_dir

Arguments:
    --file_list: Path to the text file containing the list of substrings to match in the file names.
    --input_dir: Path to the base input directory containing the dataset folders.
    --output_dir: Path to the base output directory where the filtered dataset folders will be created.
"""

# Function to create subset of files and annotations
def create_subset(original_dir, subset_dir, file_list):
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)
    
    # Filter and copy files from original to subset directory
    for root, _, files in os.walk(original_dir):
        for file in files:
            if any(substring in file for substring in file_list):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(subset_dir, os.path.relpath(src_file, original_dir))
                if not os.path.exists(os.path.dirname(dest_file)):
                    os.makedirs(os.path.dirname(dest_file))
                shutil.copy2(src_file, dest_file)
    
    # Update annotations.json
    original_annotations_path = os.path.join(original_dir, "annotations.json")
    subset_annotations_path = os.path.join(subset_dir, "annotations.json")
    if os.path.exists(original_annotations_path):
        with open(original_annotations_path, 'r') as f:
            annotations = json.load(f)
        
        subset_annotations = {
            "images": [],
            "annotations": [],
            "categories": annotations.get("categories", [])
        }
        
        # Create a list of file names to look up
        print(f"File list: {file_list}")
        
        # Create a mapping of file names to image IDs
        file_name_to_image_id = {}
        for image in annotations["images"]:
            if any(substring in image["file_name"] for substring in file_list):
                subset_annotations["images"].append(image)
                file_name_to_image_id[image["file_name"]] = image["id"]
        
        print(f"Found {len(subset_annotations['images'])} images in file list.")
        
        for annotation in annotations["annotations"]:
            if annotation["image_id"] in file_name_to_image_id.values():
                subset_annotations["annotations"].append(annotation)
        
        print(f"Found {len(subset_annotations['annotations'])} annotations matching the images.")
        
        # Write the subset annotations to the JSON file only once
        with open(subset_annotations_path, 'w') as f:
            json.dump(subset_annotations, f, indent=4)
        
        print(f"Updated annotations written to {subset_annotations_path}")

def main():
    parser = argparse.ArgumentParser(description="Create a subset of files and annotations based on a list of file name substrings.")
    parser.add_argument('--file_list', required=True, help="Path to the text file containing the list of substrings to match in the file names.")
    parser.add_argument('--input_dir', required=True, help="Path to the base input directory containing the dataset folders.")
    parser.add_argument('--output_dir', required=True, help="Path to the base output directory where the filtered dataset folders will be created.")
    
    args = parser.parse_args()

    # Read the file list
    with open(args.file_list, 'r') as f:
        file_list = [line.strip() for line in f]

    # Input and output directories
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Folders to be processed
    folders = [
        'boost_all', 'noboost_all', 'noboost_full', 'boost_full',
        'noboost_partial', 'boost_partial'
    ]

    for folder in folders:
        for subset in ['test', 'train', 'val']:
            original_dir = os.path.join(input_dir, folder, subset)
            subset_dir = os.path.join(output_dir, folder + '_SF', subset)
            print(f"Processing {original_dir} -> {subset_dir}")
            create_subset(original_dir, subset_dir, file_list)

    print("Subset creation completed.")

if __name__ == "__main__":
    main()
