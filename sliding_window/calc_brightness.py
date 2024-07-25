import cv2
import numpy as np
import json
import os
import argparse

"""
Script to extract overall average RGB values and standard deviations from the pixels within all bounding boxes in a dataset.

Usage:
    python extract_bbox_rgb_stats.py --dataset_path path_to_your_dataset --output_path output.json

Arguments:
    --dataset_path: Path to the dataset directory. The directory should contain `test`, `train`, and `val` subdirectories, each containing `annotations` and `images` subdirectories.
    --output_path: Path to the output JSON file where the results will be saved.

COCO Annotation JSON Structure:
    Each annotation JSON file should follow the COCO format:
    {
        "images": [
            {"id": int, "file_name": str, ...},
            ...
        ],
        "annotations": [
            {"image_id": int, "bbox": [x, y, width, height], ...},
            ...
        ],
        ...
    }
"""

def extract_bbox_pixels(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

def process_annotations(annotations_file, images_dir):
    all_pixels = []
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create a mapping from image ID to file name
    image_id_to_filename = {image['id']: image['file_name'] for image in coco_data['images']}
    
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        image_file = os.path.join(images_dir, image_id_to_filename[image_id])
        image = cv2.imread(image_file)
        
        if image is not None:
            bbox_pixels = extract_bbox_pixels(image, bbox)
            all_pixels.append(bbox_pixels)
    
    return all_pixels

def process_dataset(dataset_path):
    all_pixels = []
    subsets = ['test', 'train', 'val']
    
    for subset in subsets:
        annotations_path = os.path.join(dataset_path, subset, 'annotations')
        images_path = os.path.join(dataset_path, subset, 'images')
        
        for file in os.listdir(annotations_path):
            if file.endswith(".json"):
                subset_pixels = process_annotations(os.path.join(annotations_path, file), images_path)
                all_pixels.extend(subset_pixels)
    
    all_pixels = np.concatenate([pixels.reshape(-1, 3) for pixels in all_pixels], axis=0)
    mean_rgb = np.mean(all_pixels, axis=0)
    std_rgb = np.std(all_pixels, axis=0)
    
    return mean_rgb, std_rgb

def main():
    parser = argparse.ArgumentParser(description='Extract overall RGB stats from bounding boxes in a dataset')
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset directory')
    parser.add_argument('--output_path', required=True, help='Path to the output JSON file')
    
    args = parser.parse_args()
    
    mean_rgb, std_rgb = process_dataset(args.dataset_path)
    
    result = {
        'mean_rgb': mean_rgb.tolist(),
        'std_rgb': std_rgb.tolist()
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"Extraction completed. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
