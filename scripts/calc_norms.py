"""
This script calculates the mean and standard deviation of the pixel values for a random selection of 100 images from
each specified subdirectory (train, test, val) within a given directory. It reads images in color (BGR format), 
extracts pixel values for each channel (Red, Green, and Blue), and computes the mean and standard deviation for each channel.

Usage:
    python calc_norms.py <image_directory>

Arguments:
    <image_directory>: Directory containing the subdirectories train, test, and val.

Output:
    Prints the mean and standard deviation of the pixel values for each color channel (Red, Green, Blue).

Note:
    This may take a while to run

Expected Directory Tree:
.
├── test
├── train
└── val

All of the images are stored directly within train, test, and val subdirectories.
"""

import os
import sys
import numpy as np
import cv2
import random

def calculate_mean_and_std(image_dir, num_images=100):
    pixel_values_r = []
    pixel_values_g = []
    pixel_values_b = []

    subdirs = ['train', 'test', 'val']

    for subdir in subdirs:
        subdir_path = os.path.join(image_dir, subdir)
        if not os.path.isdir(subdir_path):
            print(f"Subdirectory {subdir_path} does not exist.")
            continue

        all_images = [img for img in os.listdir(subdir_path) if img.endswith('.png')]
        selected_images = random.sample(all_images, min(num_images, len(all_images)))

        for image_name in selected_images:
            image_path = os.path.join(subdir_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read in color (BGR)

            if image is None:
                print(f"Failed to read image {image_path}. Skipping.")
                continue

            pixel_values_r.extend(image[:, :, 2].flatten())  # Red channel
            pixel_values_g.extend(image[:, :, 1].flatten())  # Green channel
            pixel_values_b.extend(image[:, :, 0].flatten())  # Blue channel

    mean_r = np.mean(pixel_values_r)
    std_r = np.std(pixel_values_r)
    mean_g = np.mean(pixel_values_g)
    std_g = np.std(pixel_values_g)
    mean_b = np.mean(pixel_values_b)
    std_b = np.std(pixel_values_b)

    means_bgr = (mean_b, mean_g, mean_r)
    stds_bgr = (std_b, std_g, std_r)

    means_rgb = (mean_r, mean_g, mean_b)
    stds_rgb = (std_r, std_g, std_b)
    
    return means_bgr, stds_bgr, means_rgb, stds_rgb

def main():
    if len(sys.argv) != 2:
        print("Usage: python calc_norms.py <image_directory>")
        sys.exit(1)

    image_directory = sys.argv[1]
    
    if not os.path.isdir(image_directory):
        print(f"The path {image_directory} is not a valid directory.")
        sys.exit(1)

    means_bgr, stds_bgr, means_rgb, stds_rgb = calculate_mean_and_std(image_directory)
    means_rounded_rgb = [round(mean, 2) for mean in means_rgb]
    stds_rounded_rgb = [round(std, 2) for std in stds_rgb]
    
    print(f"normalization_values = {{")
    print(f"    'mean': {means_rounded_rgb},")
    print(f"    'std': {stds_rounded_rgb}")
    print(f"}}")

if __name__ == "__main__":
    main()
