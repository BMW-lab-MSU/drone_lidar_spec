#!/bin/bash

# RUN THIS SCRIPT FROM /home/d86p233/Desktop/BMW-spec/scripts

# Define the URL and the path where you want to save the weights
URL="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth"
SAVE_PATH="../mmdetection/checkpoints/resnext101_32x8d-110c445d.pth"

# Ensure the directory exists
mkdir -p $(dirname $SAVE_PATH)

# Download the file using wget
wget $URL -O $SAVE_PATH

# Print a success message
echo "Model weights downloaded and saved as '$SAVE_PATH'"
