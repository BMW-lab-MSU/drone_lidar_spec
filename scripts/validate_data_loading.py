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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import mmcv
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Validate Data Loading in MMDetection')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--output_dir', type=str, default='output_images', help='Directory to save output images')
    parser.add_argument('--display_images', action='store_true', help='Display images instead of saving to files')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='Override settings in the config file')
    args = parser.parse_args()
    return args

class CustomDataset(Dataset):
    def __init__(self, ann_file, img_prefix, pipeline):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.data_infos = mmcv.load(ann_file)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img_path = osp.join(self.img_prefix, data_info['filename'])
        img = mmcv.imread(img_path)
        gt_bboxes = torch.tensor(data_info['ann']['bboxes'], dtype=torch.float32)
        gt_labels = torch.tensor(data_info['ann']['labels'], dtype=torch.long)
        
        data = {
            'img': img,
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'img_metas': {'filename': data_info['filename']}
        }
        
        # Apply pipeline transformations
        for transform in self.pipeline:
            data = transform(data)
        
        return data

def main():
    args = parse_args()

    # Load the configuration
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.output_dir is not None and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    try:
        val_pipeline = cfg.data.val.pipeline
        val_ann_file = cfg.data.val.ann_file
        val_img_prefix = cfg.data.val.img_prefix
    except AttributeError as e:
        print(f"Configuration error: {e}")
        return

    # Build the validation dataset directly
    val_dataset = CustomDataset(val_ann_file, val_img_prefix, val_pipeline)

    # Create DataLoader manually
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    # Get a batch of data
    for i, data in enumerate(val_dataloader):
        if i >= 5:  # Print only the first 5 samples
            break
        img = data['img'][0]
        gt_bboxes = data['gt_bboxes'][0]
        gt_labels = data['gt_labels'][0]
        
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
