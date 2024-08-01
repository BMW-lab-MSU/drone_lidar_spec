import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

"""
This script creates EDA plots of grayscale color difference vs. tilt angle from JSON data.

Usage:
    python eda_plot.py --input data.json --output plot.png

Arguments:
    --input  Path to the input JSON file containing the data array.
    --output Path to the output PNG file for the plot.
"""

def rgb_to_grayscale(rgb):
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]

def create_eda_plot(input_file, output_file):
    # Define target RGB and convert to grayscale
    target_rgb = [146.2918103158602, 205.90643800403225, 72.32910836693549]
    target_grayscale = rgb_to_grayscale(target_rgb)

    # Read the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract tilt angle and grayscale color difference
    tilt_angles = []
    grayscale_differences = []
    
    for item in data:
        tilt_angle = item['tilt_angle']
        mean_rgb = item['best_window']['mean_rgb']
        mean_grayscale = rgb_to_grayscale(mean_rgb)
        grayscale_difference = abs(mean_grayscale - target_grayscale)
        
        tilt_angles.append(tilt_angle)
        grayscale_differences.append(grayscale_difference)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(tilt_angles, grayscale_differences, alpha=0.7)
    plt.title('Prop-Only: Grayscale Color Difference vs. Tilt Angle', fontsize=16)
    plt.xlabel('Tilt Angle (degrees)', fontsize=14)
    plt.ylabel('Grayscale Color Difference', fontsize=14)
    plt.grid(True)
    plt.xticks(np.arange(min(tilt_angles), max(tilt_angles) + 1, 10))  # Set tick marks every 10 units on the x-axis

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Plot saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create EDA plots of grayscale color difference vs. tilt angle.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output PNG file for the plot.')

    args = parser.parse_args()

    create_eda_plot(args.input, args.output)
