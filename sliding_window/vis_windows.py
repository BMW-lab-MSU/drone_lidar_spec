import cv2
import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
Script to read a JSON file and a folder with images, then plot the images with the best window and ground truth bounding boxes overlaid.

Usage:
    python vis_windows.py --json_path path_to_json_file --images_path path_to_images_folder --annotations_path path_to_annotations_file --output_dir path_to_output_directory

Arguments:
    --json_path: Path to the JSON file with window details.
    --images_path: Path to the folder containing the images.
    --annotations_path: Path to the annotations file in COCO format.
    --output_dir: Path to the output directory for saving plotted images.
"""

def load_coco_annotations(annotations_path):
    with open(annotations_path, 'r') as file:
        coco_data = json.load(file)
    return coco_data

def get_annotations_for_image(coco_data, image_file):
    image_id = next((img['id'] for img in coco_data['images'] if img['file_name'] == image_file), None)
    if image_id is None:
        return []

    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    return annotations

def plot_image_with_windows(image_path, window_details, annotations, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    window_position = window_details["window_position"]
    window_dimensions = window_details["window_dimensions"]

    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a plot
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(image_rgb)
    ax.set_title(f"{os.path.basename(image_path)}")
    ax.axis('off')

    # Draw the predicted rectangle on the image
    x, y = window_position
    w, h = window_dimensions
    predicted_rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='purple', facecolor='none')
    ax.add_patch(predicted_rect)

    # Draw the ground truth rectangles on the image
    for ann in annotations:
        x_gt, y_gt, w_gt, h_gt = map(int, ann['bbox'])
        ground_truth_rect = patches.Rectangle((x_gt, y_gt), w_gt, h_gt, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(ground_truth_rect)

        # Calculate and draw overlap if exists
        overlap_x1 = max(x, x_gt)
        overlap_y1 = max(y, y_gt)
        overlap_x2 = min(x + w, x_gt + w_gt)
        overlap_y2 = min(y + h, y_gt + h_gt)
        
        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            overlap_rect = patches.Rectangle((overlap_x1, overlap_y1), overlap_x2 - overlap_x1, overlap_y2 - overlap_y1, 
                                             linewidth=0, edgecolor='none', facecolor='orange', alpha=0.5)
            ax.add_patch(overlap_rect)

    # Save the plot
    output_image_path = os.path.join(output_dir, f"plotted_{os.path.basename(image_path)}")
    plt.savefig(output_image_path)
    plt.close()
    print(f"Saved plotted image: {output_image_path}")

def process_json(json_path, images_path, annotations_path, output_dir):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    coco_data = load_coco_annotations(annotations_path)

    for item in data:
        image_file = item["image"]
        image_path = os.path.join(images_path, image_file)
        best_window = item["best_window"]
        annotations = get_annotations_for_image(coco_data, image_file)

        plot_image_with_windows(image_path, best_window, annotations, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Plot images with windows from JSON and ground truth from COCO annotations')
    parser.add_argument('--json_path', required=True, help='Path to the JSON file with window details')
    parser.add_argument('--images_path', required=True, help='Path to the folder containing the images')
    parser.add_argument('--annotations_path', required=True, help='Path to the annotations file in COCO format')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory for saving plotted images')

    args = parser.parse_args()

    print(f"Processing JSON file: {args.json_path}")
    process_json(args.json_path, args.images_path, args.annotations_path, args.output_dir)
    print(f"Processing completed. Plotted images saved to {args.output_dir}")

if __name__ == "__main__":
    main()
