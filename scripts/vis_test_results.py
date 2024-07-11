"""
This script visualizes testing results by overlaying ground truth and predicted bounding boxes on test images.

Purpose:
---------
The purpose of this script is to:
1. Load testing results and annotations.
2. Visualize and save images with ground truth and predicted bounding boxes.
3. Compute and print evaluation metrics such as precision and recall.

Usage:
------
1. Ensure you have a pickle file containing the results from your model's inference. This can be generated using the mmdetection test script.
2. Have a directory containing the test images and an 'annotations.json' file with the ground truth annotations in COCO format.
3. Run the script with the following command:
   python visualize_results.py --results_file path/to/results.pkl --images_dir path/to/images_dir --output_dir path/to/output_dir

Arguments:
----------
--results_file: Path to the output pickle file containing results.
--images_dir: Path to the directory containing the test images.
--output_dir: Path to the directory to save the visualized images.

Example:
--------
python visualize_results.py --results_file results.pkl --images_dir ./test_images --output_dir ./output_visualizations

Example Input Directory Structure:
----------------------------------
input_base_path/
    ├── annotations.json
    ├── image1.png
    ├── image2.png
    └── ... (more image files)

Example Output Directory Structure:
-----------------------------------
output_base_path/
    ├── 1/
    │   ├── original.png
    │   ├── ground_truth.png
    │   ├── predicted.png
    │   └── combined.png
    ├── 2/
    │   ├── original.png
    │   ├── ground_truth.png
    │   ├── predicted.png
    │   └── combined.png
    └── ... (more subdirectories for each image)
"""

import argparse
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize testing results with ground truth and predicted bounding boxes.')
    parser.add_argument('--results_file', required=True, help='Path to the output pickle file containing results')
    parser.add_argument('--images_dir', required=True, help='Path to the directory containing the test images')
    parser.add_argument('--output_dir', required=True, help='Path to the directory to save the visualized images')
    return parser.parse_args()

def load_results(results_file):
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results

def load_annotations(images_dir):
    annotations_file = os.path.join(images_dir, 'annotations.json')
    with open(annotations_file, 'r') as f:
        coco_annotations = json.load(f)

    annotations = {}
    for image_info in coco_annotations['images']:
        image_id = str(image_info['id'])
        annotations[image_id] = {'bboxes': []}

    for ann in coco_annotations['annotations']:
        image_id = str(ann['image_id'])
        bbox = ann['bbox']  # COCO format: [x, y, width, height]
        annotations[image_id]['bboxes'].append(bbox)

    return annotations

def save_image(image_path, bboxes, output_path, bbox_color, title):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='viridis')
    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor=bbox_color, facecolor='none')
        ax.add_patch(rect)
    plt.title(title)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def save_combined_image(image_path, ground_truths, predictions, output_path):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='viridis')
    # Plot ground truth bounding boxes
    for bbox in ground_truths:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='#FFA500', facecolor='none')
        ax.add_patch(rect)
    # Plot predicted bounding boxes and shade overlapping regions
    for bbox in predictions:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='#FF0000', facecolor='none')
        ax.add_patch(rect)
        overlap = find_overlap(ground_truths, bbox)
        if overlap:
            for ov in overlap:
                ax.add_patch(patches.Polygon(ov, closed=True, color='#FF00FF', alpha=0.3))
    plt.title('Image with Ground Truth and Predicted Bboxes')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def find_overlap(ground_truths, pred_bbox):
    overlaps = []
    for gt_bbox in ground_truths:
        x1 = max(gt_bbox[0], pred_bbox[0])
        y1 = max(gt_bbox[1], pred_bbox[1])
        x2 = min(gt_bbox[0] + gt_bbox[2], pred_bbox[0] + pred_bbox[2])
        y2 = min(gt_bbox[1] + gt_bbox[3], pred_bbox[1] + pred_bbox[3])
        if x1 < x2 and y1 < y2:
            overlaps.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return overlaps if overlaps else None

def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def compute_metrics(results, annotations, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = 0

    for image_info in results:
        image_id = str(image_info['img_id'])
        ground_truths = annotations[image_id]['bboxes']
        predictions = image_info['pred_instances']['bboxes'].numpy()

        matched_gt = set()
        for pred_bbox in predictions:
            match_found = False
            for gt_idx, gt_bbox in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(gt_bbox, pred_bbox)
                if iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(gt_idx)
                    match_found = True
                    break
            if not match_found:
                fp += 1

        fn += len(ground_truths) - len(matched_gt)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall

def main():
    args = parse_args()
    results = load_results(args.results_file)
    annotations = load_annotations(args.images_dir)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Debugging: Print the structure of one entry in results
    if results:
        print("Example entry in results:", results[0])
    
    # Compute metrics
    precision, recall = compute_metrics(results, annotations)
    print(f'Precision: {precision}, Recall: {recall}')

    # Iterate over results and visualize
    for image_info in results:
        # Debugging: Print the keys in image_info
        print("Keys in image_info:", image_info.keys())
        
        image_id = str(image_info['img_id'])
        image_path = os.path.join(args.images_dir, os.path.basename(image_info['img_path']))
        output_subdir = os.path.join(args.output_dir, image_id)
        os.makedirs(output_subdir, exist_ok=True)

        ground_truths = annotations[image_id]['bboxes']
        predictions = image_info['pred_instances']['bboxes'].numpy()

        save_image(image_path, [], os.path.join(output_subdir, 'original.png'), bbox_color='white', title='Original Image')
        save_image(image_path, ground_truths, os.path.join(output_subdir, 'ground_truth.png'), bbox_color='#FFA500', title='Ground Truth Bboxes')
        save_image(image_path, predictions, os.path.join(output_subdir, 'predicted.png'), bbox_color='#FF0000', title='Predicted Bboxes')
        save_combined_image(image_path, ground_truths, predictions, os.path.join(output_subdir, 'combined.png'))

if __name__ == '__main__':
    main()
