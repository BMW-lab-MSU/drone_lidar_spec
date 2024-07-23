"""
This script loads a given configuration file (including all its dependencies),
prints the merged configuration, and saves it to an output file.

Usage:
    python path/to/this_script.py path/to/config.py --output path/to/merged_config.txt

Arguments:
    config  : Path to the config file to be loaded and merged.
    --output: Path to save the merged configuration file.

Example:
    python print_config.py /home/d86p233/Desktop/BMW-spec/mmdetection/configs/faster_rcnn/faster-rcnn_x101-32x8d_fpn_ms-3x_coco.py --output /home/d86p233/Desktop/BMW-spec/merged_config.txt
"""

import os
from argparse import ArgumentParser
from mmengine.config import Config

def parse_args():
    parser = ArgumentParser(description='Load and print a merged configuration file.')
    parser.add_argument('config', type=str, help='Path to the config file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the merged configuration file.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load and merge configuration
    cfg = Config.fromfile(args.config)
    
    # Print merged configuration
    print(f'Using configuration:\n{cfg.pretty_text}')
    
    # Save merged configuration to output file
    with open(args.output, 'w') as f:
        f.write(cfg.pretty_text)
    
    print(f'Merged configuration saved to {args.output}')

if __name__ == '__main__':
    main()
