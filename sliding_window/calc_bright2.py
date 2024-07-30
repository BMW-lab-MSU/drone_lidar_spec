import os
import json
import numpy as np
from PIL import Image
import argparse

"""
This script calculates the average and standard deviation of RGB values for bounding boxes 
in COCO-format annotations and saves the results as a JSON file.

How to Use:

1. Ensure the directory structure:
   .
   |-- annotations
   |   `-- annotations.json
   `-- images
       |-- image1.png
       |-- image2.png
       `-- ...

2. Install required packages:
   Ensure you have the necessary Python packages installed. You can do this using:
   pip install numpy pillow

3. Run the script:
   Execute the script using Python with the appropriate command-line arguments:
   python calculate_avg_rgb.py --annotations_path annotations/annotations.json --images_path images --output_path results.json

4. Output:
   The script will create a results.json file containing the mean and standard deviation 
   of the RGB values of the bounding boxes.

Example of Output JSON (results.json):
{
    "mean_rgb": [71.60, 206.33, 147.64],
    "std_rgb": [10.50, 15.20, 12.30]
}
"""

def calculate_rgb_statistics(window):
    if window.shape[2] == 4:
        window = window[:, :, :3]  # Convert RGBA to RGB if necessary
    mean_rgb = np.mean(window, axis=(0, 1))
    std_rgb = np.std(window, axis=(0, 1))
    return mean_rgb, std_rgb

def calculate_avg_rgb(annotations_path, images_path, output_path):
    # Load the annotations JSON file
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)

    # Initialize separate lists to hold RGB values
    all_r_values = []
    all_g_values = []
    all_b_values = []
    all_r_std = []
    all_g_std = []
    all_b_std = []

    # Process each annotation
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']  # [x, y, width, height]

        # Find the corresponding image file
        image_info = next(image for image in annotations['images'] if image['id'] == image_id)
        image_path = os.path.join(images_path, image_info['file_name'])
        
        # Load the image
        img = Image.open(image_path)
        img_array = np.array(img)

        # Extract the bounding box region
        x, y, width, height = map(int, bbox)
        bbox_region = img_array[y:y+height, x:x+width]

        # Calculate the RGB statistics for the bounding box
        mean_rgb, std_rgb = calculate_rgb_statistics(bbox_region)
        all_r_values.append(mean_rgb[0])
        all_g_values.append(mean_rgb[1])
        all_b_values.append(mean_rgb[2])
        all_r_std.append(std_rgb[0])
        all_g_std.append(std_rgb[1])
        all_b_std.append(std_rgb[2])

    # Calculate the overall statistics
    overall_avg_rgb = [np.mean(all_r_values), np.mean(all_g_values), np.mean(all_b_values)]
    overall_std_rgb = [np.mean(all_r_std), np.mean(all_g_std), np.mean(all_b_std)]

    # Save the results to a JSON file
    results = {
        'mean_rgb': overall_avg_rgb,
        'std_rgb': overall_std_rgb
    }
    with open(output_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Calculate the average and standard deviation of RGB values for bounding boxes in COCO-format annotations.')
    parser.add_argument('--annotations_path', type=str, required=True, help='Path to the annotations JSON file')
    parser.add_argument('--images_path', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON file')

    args = parser.parse_args()

    calculate_avg_rgb(args.annotations_path, args.images_path, args.output_path)

if __name__ == "__main__":
    main()
