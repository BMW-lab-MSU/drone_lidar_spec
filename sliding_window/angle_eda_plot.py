import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict

"""
This script creates EDA plots of squared error vs. tilt angle from JSON data, including:
1. A scatter plot of squared error vs. tilt angle.
2. A bar plot showing the mean squared error at each tilt angle with error bars representing the standard deviation.
3. A bar plot showing the standard deviation of squared error at each tilt angle.
Both log-transformed and non-log-transformed plots are included, making a total of 6 plots.

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
    
    # Extract tilt angle and squared error
    tilt_angles = [item['tilt_angle'] for item in data]
    squared_errors = [item['squared_error'] for item in data]
    
    # Apply log transformation to squared errors
    log_squared_errors = np.log1p(squared_errors)  # log1p handles log(0) issues
    
    # Prepare data for mean and standard deviation plot
    error_by_angle = defaultdict(list)
    log_error_by_angle = defaultdict(list)
    
    for angle, error, log_error in zip(tilt_angles, squared_errors, log_squared_errors):
        error_by_angle[angle].append(error)
        log_error_by_angle[angle].append(log_error)
    
    angles = sorted(error_by_angle.keys())
    means = [np.mean(error_by_angle[angle]) for angle in angles]
    std_devs = [np.std(error_by_angle[angle]) for angle in angles]
    
    log_means = [np.mean(log_error_by_angle[angle]) for angle in angles]
    log_std_devs = [np.std(log_error_by_angle[angle]) for angle in angles]

    # Create the subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))

    # Top left subplot: scatter plot (non-log-transformed)
    axs[0, 0].scatter(tilt_angles, squared_errors, alpha=0.7)
    axs[0, 0].set_title('Squared Error vs. Tilt Angle')
    axs[0, 0].set_xlabel('Tilt Angle (degrees)')
    axs[0, 0].set_ylabel('Squared Error')
    axs[0, 0].grid(True)

    # Top right subplot: scatter plot (log-transformed)
    axs[0, 1].scatter(tilt_angles, log_squared_errors, alpha=0.7)
    axs[0, 1].set_title('Log Transformed Squared Error vs. Tilt Angle')
    axs[0, 1].set_xlabel('Tilt Angle (degrees)')
    axs[0, 1].set_ylabel('Log Transformed Squared Error')
    axs[0, 1].grid(True)

    # Middle left subplot: mean and std deviation plot (non-log-transformed)
    axs[1, 0].errorbar(angles, means, yerr=std_devs, fmt='o', ecolor='r', capsize=5)
    axs[1, 0].set_title('Mean Squared Error vs. Tilt Angle with Error Bars')
    axs[1, 0].set_xlabel('Tilt Angle (degrees)')
    axs[1, 0].set_ylabel('Mean Squared Error')
    axs[1, 0].grid(True)

    # Middle right subplot: mean and std deviation plot (log-transformed)
    axs[1, 1].errorbar(angles, log_means, yerr=log_std_devs, fmt='o', ecolor='r', capsize=5)
    axs[1, 1].set_title('Mean Log Transformed Squared Error vs. Tilt Angle with Error Bars')
    axs[1, 1].set_xlabel('Tilt Angle (degrees)')
    axs[1, 1].set_ylabel('Mean Log Transformed Squared Error')
    axs[1, 1].grid(True)

    # Bottom left subplot: standard deviation plot (non-log-transformed)
    axs[2, 0].bar(angles, std_devs, color='skyblue', alpha=0.7)
    axs[2, 0].set_title('Standard Deviation of Squared Error vs. Tilt Angle')
    axs[2, 0].set_xlabel('Tilt Angle (degrees)')
    axs[2, 0].set_ylabel('Standard Deviation of Squared Error')
    axs[2, 0].grid(True)

    # Bottom right subplot: standard deviation plot (log-transformed)
    axs[2, 1].bar(angles, log_std_devs, color='skyblue', alpha=0.7)
    axs[2, 1].set_title('Standard Deviation of Log Transformed Squared Error vs. Tilt Angle')
    axs[2, 1].set_xlabel('Tilt Angle (degrees)')
    axs[2, 1].set_ylabel('Standard Deviation of Log Transformed Squared Error')
    axs[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Plot saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create EDA plots of squared error vs. tilt angle.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output PNG file for the plot.')

    args = parser.parse_args()

    create_eda_plot(args.input, args.output)
