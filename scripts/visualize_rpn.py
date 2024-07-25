"""
This script loads a pre-trained object detection model, runs inference to extract Region Proposal Network (RPN) proposals, and visualizes the proposals on a batch of input images.

Usage:
    python visualize_rpn.py --config_file path/to/config.py --checkpoint_file path/to/checkpoint.pth --image_folder path/to/images/ --batch_size 8 --num_images 10 --device cpu

Arguments:
    --config_file: Path to the model configuration file.
    --checkpoint_file: Path to the model checkpoint file.
    --image_folder: Path to the folder containing images for inference.
    --batch_size: Number of images to process in each batch.
    --num_images: Number of images to process from the input folder.
    --device: Device to run the inference on (cpu or cuda:0).
"""

import os
import argparse
from mmdet.apis import init_detector
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_rpn_proposals(model, imgs, device):
    proposals_batch = []
    
    for img in imgs:
        img_data = model.data_preprocessor(img, return_raw=True)
        img_tensor = img_data['inputs'][0].unsqueeze(0).to(device)

        with torch.no_grad():
            x = model.extract_feat(img_tensor)
            rpn_outs = model.rpn_head(x)
            rpn_proposals = model.rpn_head.get_bboxes(*rpn_outs, img_metas=[{'ori_shape': img.shape[:2], 'img_shape': img.shape[:2], 'pad_shape': img.shape[:2], 'scale_factor': 1.0}])
        
        proposals_batch.append(rpn_proposals[0])
    
    return proposals_batch

def visualize_proposals(img_paths, proposals_batch, model):
    for img_path, proposals in zip(img_paths, proposals_batch):
        img = mmcv.imread(img_path)
        img = mmcv.imdenormalize(img, mean=model.data_preprocessor.mean, std=model.data_preprocessor.std, to_bgr=model.data_preprocessor.bgr_to_rgb)

        proposals = proposals.cpu().numpy()
        mmcv.imshow_bboxes(img, proposals, show=True)

def main(config_file, checkpoint_file, image_folder, batch_size, num_images, device):
    model = init_detector(config_file, checkpoint_file, device=device)

    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))][:num_images]
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    for batch in batches:
        imgs = [mmcv.imread(img_path) for img_path in batch]
        proposals_batch = get_rpn_proposals(model, imgs, device)
        visualize_proposals(batch, proposals_batch, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a pre-trained model and visualize RPN proposals on a batch of images.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument("--checkpoint_file", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images for inference")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process in each batch")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to process from the input folder")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to run the inference on (cpu or cuda:0)")
    
    args = parser.parse_args()
    main(args.config_file, args.checkpoint_file, args.image_folder, args.batch_size, args.num_images, args.device)
