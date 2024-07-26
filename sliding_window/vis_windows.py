import cv2
import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt

"""
Script to read a JSON file and a folder with images, then plot the images with the best window overlaid.

Usage:
    python vis_windows.py --json_path path_to_json_file --images_path path_to_images_folder --output_dir path_to_output_directory

Arguments:
    --json_path: Path to the JSON file with window details.
    --images_path: Path to the folder containing the images.
    --output_dir: Path to the output directory for saving plotted images.
"""

def plot_image_with_window(image_path, window_details, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    window_position = window_details["window_position"]
    window_dimensions = window_details["window_dimensions"]

    # Draw the rectangle on the image
    x, y = window_position
    w, h = window_dimensions
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plotting the image
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title(f"{os.path.basename(image_path)}")
    plt.axis('off')

    # Save the plot
    output_image_path = os.path.join(output_dir, f"plotted_{os.path.basename(image_path)}")
    plt.savefig(output_image_path)
    plt.close()
    print(f"Saved plotted image: {output_image_path}")

def process_json(json_path, images_path, output_dir):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for item in data:
        image_file = item["image"]
        image_path = os.path.join(images_path, image_file)
        best_window = item["best_window"]

        plot_image_with_window(image_path, best_window, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Plot images with windows from JSON')
    parser.add_argument('--json_path', required=True, help='Path to the JSON file with window details')
    parser.add_argument('--images_path', required=True, help='Path to the folder containing the images')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory for saving plotted images')

    args = parser.parse_args()

    print(f"Processing JSON file: {args.json_path}")
    process_json(args.json_path, args.images_path, args.output_dir)
    print(f"Processing completed. Plotted images saved to {args.output_dir}")

if __name__ == "__main__":
    main()
