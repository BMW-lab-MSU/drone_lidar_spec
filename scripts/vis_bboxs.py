"""
Script: visualize_ground_truths.py

Description:
    This script visualizes ground truth bounding boxes on images. It overlays the bounding boxes on the images and saves the resulting images to a specified output directory.

Usage:
    python visualize_ground_truths.py --images_dir <path_to_images_directory> --output_dir <path_to_output_directory> --num_images <number_of_images_to_visualize>

Arguments:
    --images_dir     : Path to the directory containing the images and the annotations.json file.
    --output_dir     : Path to the directory to save the visualized images.
    --num_images     : Number of images to visualize (default is 10).

Example:
    python visualize_ground_truths.py --images_dir /path/to/images_dir --output_dir /path/to/output_dir --num_images 10
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize ground truth bounding boxes on images.')
    parser.add_argument('--images_dir', required=True, help='Path to the directory containing the images and annotations')
    parser.add_argument('--output_dir', required=True, help='Path to the directory to save the visualized images')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to visualize')
    return parser.parse_args()

def load_annotations(images_dir):
    annotations_file = os.path.join(images_dir, 'annotations.json')
    with open(annotations_file, 'r') as f:
        coco_annotations = json.load(f)

    annotations = {}
    for image_info in coco_annotations['images']:
        image_id = str(image_info['id'])
        annotations[image_id] = {
            'file_name': image_info['file_name'],
            'bboxes': []
        }

    for ann in coco_annotations['annotations']:
        image_id = str(ann['image_id'])
        bbox = ann['bbox']  # COCO format: [x, y, width, height]
        annotations[image_id]['bboxes'].append(bbox)

    return annotations

def visualize(image_path, bboxes, output_path):
    # Load image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(7.75, 4.62))
    ax.imshow(image, cmap='viridis')
    
    # Plot ground truth bounding boxes
    for bbox in bboxes:
        print(f"Plotting bbox: {bbox}")  # Debugging: Print the bounding box being plotted
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='#FFA500', facecolor='none')
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    args = parse_args()
    annotations = load_annotations(args.images_dir)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Visualize specified number of images
    count = 0
    for image_id, data in annotations.items():
        if count >= args.num_images:
            break
        print(f"Processing image ID: {image_id}")  # Debugging: Print the image ID being processed
        print(f"Bounding boxes: {data['bboxes']}")  # Debugging: Print the bounding boxes for the image
        image_path = os.path.join(args.images_dir, data['file_name'])
        output_path = os.path.join(args.output_dir, f"{os.path.splitext(data['file_name'])[0]}_gt.png")
        visualize(image_path, data['bboxes'], output_path)
        count += 1

if __name__ == '__main__':
    main()
