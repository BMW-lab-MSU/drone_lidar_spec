import torch
import sys

def load_checkpoint_and_print_shapes(checkpoint_path):
    """
    Load a PyTorch checkpoint and print the layer names and their corresponding shapes.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Example Usage:
        python script_name.py /path/to/checkpoint.pth
    """
    # Load the checkpoint file
    checkpoint = torch.load(checkpoint_path)

    # Extract the state dictionary
    state_dict = checkpoint['state_dict']

    # Print the layer names and their corresponding shapes
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")

if __name__ == "__main__":
    # Ensure a checkpoint path is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py /path/to/checkpoint.pth")
    else:
        checkpoint_path = sys.argv[1]
        load_checkpoint_and_print_shapes(checkpoint_path)
