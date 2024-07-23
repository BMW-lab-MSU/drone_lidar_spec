import torch
import argparse

"""
Script to inspect a .pth file to determine if it contains weights for just the backbone or for the whole model.
Usage: python inspect_checkpoint.py --checkpoint_path /path/to/your/checkpoint.pth
"""

def check_component(state_dict, component_name):
    """
    Check if keys for a specific component exist in the state dictionary.

    Args:
        state_dict (dict): The state dictionary containing model weights.
        component_name (str): The name of the component to check (e.g., 'backbone').

    Returns:
        bool: True if keys for the component exist, False otherwise.
    """
    for key in state_dict.keys():
        if key.startswith(component_name):
            return True
    return False

def inspect_checkpoint(checkpoint_path):
    """
    Inspect the checkpoint file to determine if it contains weights for the backbone, neck, RPN head, and ROI head.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Inspect the keys in the state dictionary
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # Sometimes the state_dict might be at the root level

    # Print the keys to see which parts of the model are included
    for key in state_dict.keys():
        print(key)

    # Check if the checkpoint includes specific components
    has_backbone = check_component(state_dict, 'conv1') or check_component(state_dict, 'layer1')
    has_neck = check_component(state_dict, 'neck')
    has_rpn_head = check_component(state_dict, 'rpn_head')
    has_roi_head = check_component(state_dict, 'roi_head')

    print(f"Has backbone: {has_backbone}")
    print(f"Has neck: {has_neck}")
    print(f"Has RPN head: {has_rpn_head}")
    print(f"Has ROI head: {has_roi_head}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect a .pth checkpoint file.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file.')

    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint_path)
