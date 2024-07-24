import os
import shutil
import sys

"""
Script to split each 'train', 'test', and 'val' folder into two folders: 'annotations' and 'images'.
It moves the respective files to these folders appropriately if they are not already organized.

How to use:
1. Save this script as `split_folders.py`.
2. Run the script using the command: `python split_folders.py /path/to/your/dataset`.

Directory structure before running the script:
dataset/
├── boost_all
│   ├── test
│   ├── train
│   └── val
├── boost_all_SF
│   ├── test
│   ├── train
│   └── val
...

Directory structure after running the script:
dataset/
├── boost_all
│   ├── test
│   │   ├── annotations
│   │   │   └── annotations.json
│   │   └── images
│   │       └── *.png
│   ├── train
│   │   ├── annotations
│   │   │   └── annotations.json
│   │   └── images
│   │       └── *.png
│   └── val
│       ├── annotations
│       │   └── annotations.json
│       └── images
│           └── *.png
├── boost_all_SF
│   ├── test
│   │   ├── annotations
│   │   │   └── annotations.json
│   │   └── images
│   │       └── *.png
│   ├── train
│   │   ├── annotations
│   │   │   └── annotations.json
│   │   └── images
│   │       └── *.png
│   └── val
│       ├── annotations
│       │   └── annotations.json
│       └── images
│           └── *.png
...
"""

def create_and_move_files(base_path, folder_name):
    # Paths for the original folder and new subfolders
    original_folder = os.path.join(base_path, folder_name)
    annotations_folder = os.path.join(original_folder, 'annotations')
    images_folder = os.path.join(original_folder, 'images')

    # Only proceed if the folders are not already split
    if not os.path.exists(annotations_folder) and not os.path.exists(images_folder):
        # Create new subfolders
        os.makedirs(annotations_folder, exist_ok=True)
        os.makedirs(images_folder, exist_ok=True)

        # Iterate over files in the original folder
        for filename in os.listdir(original_folder):
            file_path = os.path.join(original_folder, filename)
            if os.path.isfile(file_path):
                if filename.endswith('.json'):
                    shutil.move(file_path, os.path.join(annotations_folder, filename))
                elif filename.endswith(('.png', '.jpg', '.jpeg')):
                    shutil.move(file_path, os.path.join(images_folder, filename))

def main(base_path):
    # Get all subdirectories in the base path
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            # Process train, test, and val folders in each subdirectory
            for folder_name in ['train', 'test', 'val']:
                folder_path = os.path.join(subdir_path, folder_name)
                if os.path.exists(folder_path):
                    create_and_move_files(subdir_path, folder_name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_folders.py /path/to/your/dataset")
        sys.exit(1)
        
    base_path = sys.argv[1]  # Get the base path from the command line argument
    main(base_path)
