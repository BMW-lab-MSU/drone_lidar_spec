import torch
import torchvision.models as models
import os

# Specify the model
model = models.resnext101_32x8d(pretrained=True)

# Define the path where you want to save the weights
save_path = '/mmdetection/checkpoints/resnext101_32x8d-110c445d.pth'

# Ensure the directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the model weights
torch.save(model.state_dict(), save_path)

print(f"Model weights downloaded and saved as '{save_path}'")
