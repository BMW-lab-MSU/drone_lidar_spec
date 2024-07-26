import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict

"""
This script creates EDA plots of absolute error vs. tilt angle from JSON data, including:
1. A scatter plot of absolute error vs. tilt angle.
2. A bar plot showing the mean absolute error at each tilt angle with error bars representing the standard deviation.
3. A bar plot showing the standard deviation of absolute error at each tilt angle.

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
    
    # Apply log transformation to absolute errors
    log_absolute_errors = np.log1p(absolute_errors)  # log1p handles log(0) issues
    
    # Prepare data for mean and standard deviation plot
    error_by_angle = defaultdict(list)
    log_error_by_angle = defaultdict(list)
    
    for angle, error, log_error in zip(tilt_angles, absolute_errors, log_absolute_errors):
        error_by_angle[angle].append(error)
        log_error_by_angle[angle].append(log_error)
    
    angles = sorted(error_by_angle.keys())
    means = [np.mean(error_by_angle[angle]) for angle in angles]
    std_devs = [np.std(error_by_angle[angle]) for angle in angles]
    
    log_means = [np.mean(log_error_by_angle[angle]) for angle in angles]
    log_std_devs = [np.std(log_error_by_angle[angle]) for angle in angles]

    # Create the subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

    # Top subplot: scatter plot
    ax1.scatter(tilt_angles, log_absolute_errors, alpha=0.7)
    ax1.set_title('Log Transformed Absolute Error vs. Tilt Angle')
    ax1.set_xlabel('Tilt Angle (degrees)')
    ax1.set_ylabel('Log Transformed Absolute Error')
    ax1.grid(True)

    # Middle subplot: mean and std deviation plot
    ax2.errorbar(angles, log_means, yerr=log_std_devs, fmt='o', ecolor='r', capsize=5)
    ax2.set_title('Mean Log Transformed Absolute Error vs. Tilt Angle with Error Bars')
    ax2.set_xlabel('Tilt Angle (degrees)')
    ax2.set_ylabel('Mean Log Transformed Absolute Error')
    ax2.grid(True)

    # Bottom subplot: standard deviation plot
    ax3.bar(angles, log_std_devs, color='skyblue', alpha=0.7)
    ax3.set_title('Standard Deviation of Log Transformed Absolute Error vs. Tilt Angle')
    ax3.set_xlabel('Tilt Angle (degrees)')
    ax3.set_ylabel('Standard Deviation of Log Transformed Absolute Error')
    ax3.grid(True)

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
