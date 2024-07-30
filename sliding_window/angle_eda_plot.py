import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict

"""
This script creates EDA plots of absolute error vs. tilt angle from JSON data, including:
1. A scatter plot of absolute error vs. tilt angle.

Usage:
    python eda_plot.py --input data.json --output plot.png

Arguments:
    --input  Path to the input JSON file containing the data array.
    --output Path to the output PNG file for the plot.
"""

def create_eda_plot(input_file, output_file):
    # Read the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract tilt angle and absolute error
    tilt_angles = [item['tilt_angle'] for item in data]
    absolute_errors = [item['absolute_error'] for item in data]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(tilt_angles, absolute_errors, alpha=0.7)
    plt.title('Prop-Only: Absolute Error vs. Tilt Angle', fontsize=16)
    plt.xlabel('Tilt Angle (degrees)', fontsize=14)
    plt.ylabel('Absolute Error', fontsize=14)
    plt.grid(True)
    plt.xticks(np.arange(min(tilt_angles), max(tilt_angles) + 1, 10))  # Set tick marks every 10 units on the x-axis

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Plot saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create EDA plots of absolute error vs. tilt angle.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output PNG file for the plot.')

    args = parser.parse_args()

    create_eda_plot(args.input, args.output)
