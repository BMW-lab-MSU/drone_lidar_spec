"""
This script loads a pre-trained object detection model, runs inference to extract Region Proposal Network (RPN) proposals, and visualizes the proposals on a batch of input images by saving the results.

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
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample

def get_rpn_proposals(model, imgs, device):
    proposals_batch = []
    
    for img in imgs:
        # Convert image to the expected format (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # Preprocess image
        img_data = model.data_preprocessor({'inputs': img, 'data_samples': None})
        img_tensor = img_data['inputs'][0].unsqueeze(0).to(device)
        
        with torch.no_grad():
            x = model.extract_feat(img_tensor)
            rpn_outs = model.rpn_head(x)
            
            # Precompute the shapes outside the loop
            shapes = [feat.shape[-2:] for feat in x]
            priors = model.rpn_head.prior_generator.grid_priors(shapes, device=device)
            
            # Process the outputs to get the RPN proposals
            proposals = []
            for feat, anchors in zip(rpn_outs[0], priors):
                proposals.append(anchors)
            proposals = torch.cat(proposals, dim=0)
        
        proposals_batch.append(proposals)
    
    return proposals_batch

def visualize_proposals(img_paths, proposals_batch, output_folder, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_path, proposals in zip(img_paths, proposals_batch):
        img = mmcv.imread(img_path)

        proposals = proposals.cpu().numpy()
        img_with_proposals = mmcv.imshow_bboxes(img, proposals, show=False)
        
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        mmcv.imwrite(img_with_proposals, output_path)

def main(config_file, checkpoint_file, image_folder, batch_size, num_images, device):
    model = init_detector(config_file, checkpoint_file, device=device)

    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))][:num_images]
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    output_folder = os.path.join(image_folder, 'output')
    
    for batch in batches:
        imgs = [mmcv.imread(img_path) for img_path in batch]
        proposals_batch = get_rpn_proposals(model, imgs, device)
        visualize_proposals(batch, proposals_batch, output_folder, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a pre-trained model and visualize RPN proposals on a batch of images by saving the results.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument("--checkpoint_file", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images for inference")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process in each batch")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to process from the input folder")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to run the inference on (cpu or cuda:0)")
    
    args = parser.parse_args()
    main(args.config_file, args.checkpoint_file, args.image_folder, args.batch_size, args.num_images, args.device)
