"""
Script to Validate Data Loading and Annotations in MMDetection

This script is designed to validate the loading and processing of the validation dataset in MMDetection.
It prints a few samples from the validation dataset, including images and their corresponding annotations
(bounding boxes and labels), and visualizes the images with bounding boxes drawn.

Usage:
    python validate_data_loading.py --config /path/to/your/config.py

Arguments:
    --config: Path to the configuration file.
    --output_dir: Directory to save output images (default: 'output_images').
    --display_images: Display images instead of saving to files (optional).
"""

import argparse
import os
import os.path as osp
from mmengine.config import Config, DictAction
from mmengine.dataset import build_dataset, build_dataloader
import matplotlib.pyplot as plt
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(description='Validate Data Loading in MMDetection')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--output_dir', type=str, default='output_images', help='Directory to save output images')
    parser.add_argument('--display_images', action='store_true', help='Display images instead of saving to files')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='Override settings in the config file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load the configuration
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.output_dir is not None and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Build the validation dataset and dataloader
    val_dataset = build_dataset(cfg.data.val)
    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False
    )

    # Get a batch of data
    for i, data in enumerate(val_dataloader):
        if i >= 5:  # Print only the first 5 samples
            break
        img = data['img'][0].data[0]
        gt_bboxes = data['gt_bboxes'][0].data[0]
        gt_labels = data['gt_labels'][0].data[0]
        
        print(f"Image {i+1}:")
        print(f"  Image shape: {img.shape}")
        print(f"  Ground truth bboxes: {gt_bboxes}")
        print(f"  Ground truth labels: {gt_labels}")

        # Visualize the image and bounding boxes
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format
        plt.imshow(img)
        for bbox in gt_bboxes.numpy():
            x1, y1, x2, y2 = bbox
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='r', facecolor='none'))

        # Display or save the image
        if args.display_images:
            plt.show()
        else:
            output_path = osp.join(args.output_dir, f"image_{i+1}.png")
            plt.savefig(output_path)
            plt.clf()

if __name__ == "__main__":
    main()
