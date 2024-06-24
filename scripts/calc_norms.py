"""
This script calculates the mean and standard deviation of the pixel values for all images in a specified directory.
It reads images in color (BGR format), extracts pixel values for each channel (Red, Green, and Blue), and computes
the mean and standard deviation for each channel.

Usage:
    python script_name.py <image_directory>

Arguments:
    <image_directory>: Directory containing the images.

Output:
    Prints the mean and standard deviation of the pixel values for each color channel (Red, Green, Blue).
"""

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
