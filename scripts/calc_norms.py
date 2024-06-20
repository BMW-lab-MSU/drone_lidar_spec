# Spectrogram Image Normalization Calculator 📊🎶
#
# This script calculates the mean and standard deviation for each color channel (Red, Green, Blue) of spectrogram images in a directory.
# These values are essential for normalizing your images for transfer learning with models like those in mmdetection.
#
# ## Usage 🛠️
# 1. Save the script as `calculate_mean_std.py`.
# 2. Run the script with the directory of your spectrogram images:
#    ```bash
#    python calculate_mean_std.py /path/to/your/spectrogram_images
#    ```
#
# ## Output:
# The script prints the mean and standard deviation for each color channel:
#    ```bash
#    Means: (mean_r, mean_g, mean_b)
#    Standard Deviations: (std_r, std_g, std_b)
#    ```
#
# ## Example 🔍
#    ```bash
#    python calculate_mean_std.py ./spectrograms
#    ```
#    Output:
#    ```bash
#    Means: (120.34, 115.67, 123.45)
#    Standard Deviations: (60.21, 59.78, 61.34)
#    ```
#
# ## Notes 📝
# - Ensure your images are in color (RGB).
# - The script reads images in BGR format (OpenCV convention).
# - Processing time depends on the number and size of images.
#
# Happy normalizing! 🎉

import os
import sys
import numpy as np
import cv2

def calculate_mean_and_std(image_dir):
    pixel_values_r = []
    pixel_values_g = []
    pixel_values_b = []

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read in color (BGR)

        pixel_values_r.extend(image[:, :, 2].flatten())  # Red channel
        pixel_values_g.extend(image[:, :, 1].flatten())  # Green channel
        pixel_values_b.extend(image[:, :, 0].flatten())  # Blue channel

    mean_r = np.mean(pixel_values_r)
    std_r = np.std(pixel_values_r)
    mean_g = np.mean(pixel_values_g)
    std_g = np.std(pixel_values_g)
    mean_b = np.mean(pixel_values_b)
    std_b = np.std(pixel_values_b)

    means = (mean_r, mean_g, mean_b)
    stds = (std_r, std_g, std_b)
    
    return means, stds

def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <image_directory>")
        sys.exit(1)

    image_directory = sys.argv[1]
    
    if not os.path.isdir(image_directory):
        print(f"The path {image_directory} is not a valid directory.")
        sys.exit(1)

    means, stds = calculate_mean_and_std(image_directory)
    print(f"Means: {means}")
    print(f"Standard Deviations: {stds}")

if __name__ == "__main__":
    main()
