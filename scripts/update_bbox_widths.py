import os
import json
import argparse

def update_bbox_widths(base_dir):
    """
    Traverse the given directory, identify JSON files with COCO-format annotations,
    and adjust the width of each bounding box to be the width of the entire image.

    Parameters:
    base_dir (str): The base directory to start searching for JSON files.

    Returns:
    None
    """
    # Traverse the directory
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Assuming the image info is stored under 'images' and annotations under 'annotations'
                images = {img['id']: img['width'] for img in data['images']}
                annotations = data['annotations']
                
                for ann in annotations:
                    image_id = ann['image_id']
                    image_width = images[image_id]
                    ann['bbox'][2] = image_width  # Update the width of the bbox

                # Write the updated data back to the file
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=4)

                print(f"Updated {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Update bounding box widths to match image widths in COCO-format JSON files.")
    parser.add_argument('directory', type=str, help="The base directory to start searching for JSON files.")
    args = parser.parse_args()

    update_bbox_widths(args.directory)

if __name__ == "__main__":
    main()
