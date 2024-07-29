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
Both log-transformed and non-log-transformed plots are included, making a total of 6 plots.
Additionally, square root transformed plots are also included, making a total of 9 plots.

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

    # Apply square root transformation to absolute errors
    sqrt_absolute_errors = np.sqrt(absolute_errors)
    
    # Prepare data for mean and standard deviation plot
    error_by_angle = defaultdict(list)
    log_error_by_angle = defaultdict(list)
    sqrt_error_by_angle = defaultdict(list)
    
    for angle, error, log_error, sqrt_error in zip(tilt_angles, absolute_errors, log_absolute_errors, sqrt_absolute_errors):
        error_by_angle[angle].append(error)
        log_error_by_angle[angle].append(log_error)
        sqrt_error_by_angle[angle].append(sqrt_error)
    
    angles = sorted(error_by_angle.keys())
    means = [np.mean(error_by_angle[angle]) for angle in angles]
    std_devs = [np.std(error_by_angle[angle]) for angle in angles]
    
    log_means = [np.mean(log_error_by_angle[angle]) for angle in angles]
    log_std_devs = [np.std(log_error_by_angle[angle]) for angle in angles]

    sqrt_means = [np.mean(sqrt_error_by_angle[angle]) for angle in angles]
    sqrt_std_devs = [np.std(sqrt_error_by_angle[angle]) for angle in angles]

    # Create the subplots
    fig, axs = plt.subplots(3, 3, figsize=(20, 18))

    # Top left subplot: scatter plot (non-log-transformed)
    axs[0, 0].scatter(tilt_angles, absolute_errors, alpha=0.7)
    axs[0, 0].set_title('Absolute Error vs. Tilt Angle')
    axs[0, 0].set_xlabel('Tilt Angle (degrees)')
    axs[0, 0].set_ylabel('Absolute Error')
    axs[0, 0].grid(True)

    # Top middle subplot: scatter plot (log-transformed)
    axs[0, 1].scatter(tilt_angles, log_absolute_errors, alpha=0.7)
    axs[0, 1].set_title('Log Transformed Absolute Error vs. Tilt Angle')
    axs[0, 1].set_xlabel('Tilt Angle (degrees)')
    axs[0, 1].set_ylabel('Log Transformed Absolute Error')
    axs[0, 1].grid(True)

    # Top right subplot: scatter plot (sqrt-transformed)
    axs[0, 2].scatter(tilt_angles, sqrt_absolute_errors, alpha=0.7)
    axs[0, 2].set_title('Sqrt Transformed Absolute Error vs. Tilt Angle')
    axs[0, 2].set_xlabel('Tilt Angle (degrees)')
    axs[0, 2].set_ylabel('Sqrt Transformed Absolute Error')
    axs[0, 2].grid(True)

    # Middle left subplot: mean and std deviation plot (non-log-transformed)
    axs[1, 0].errorbar(angles, means, yerr=std_devs, fmt='o', ecolor='r', capsize=5)
    axs[1, 0].set_title('Mean Absolute Error vs. Tilt Angle with Error Bars')
    axs[1, 0].set_xlabel('Tilt Angle (degrees)')
    axs[1, 0].set_ylabel('Mean Absolute Error')
    axs[1, 0].grid(True)

    # Middle middle subplot: mean and std deviation plot (log-transformed)
    axs[1, 1].errorbar(angles, log_means, yerr=log_std_devs, fmt='o', ecolor='r', capsize=5)
    axs[1, 1].set_title('Mean Log Transformed Absolute Error vs. Tilt Angle with Error Bars')
    axs[1, 1].set_xlabel('Tilt Angle (degrees)')
    axs[1, 1].set_ylabel('Mean Log Transformed Absolute Error')
    axs[1, 1].grid(True)

    # Middle right subplot: mean and std deviation plot (sqrt-transformed)
    axs[1, 2].errorbar(angles, sqrt_means, yerr=sqrt_std_devs, fmt='o', ecolor='r', capsize=5)
    axs[1, 2].set_title('Mean Sqrt Transformed Absolute Error vs. Tilt Angle with Error Bars')
    axs[1, 2].set_xlabel('Tilt Angle (degrees)')
    axs[1, 2].set_ylabel('Mean Sqrt Transformed Absolute Error')
    axs[1, 2].grid(True)

    # Bottom left subplot: standard deviation plot (non-log-transformed)
    axs[2, 0].bar(angles, std_devs, color='skyblue', alpha=0.7)
    axs[2, 0].set_title('Standard Deviation of Absolute Error vs. Tilt Angle')
    axs[2, 0].set_xlabel('Tilt Angle (degrees)')
    axs[2, 0].set_ylabel('Standard Deviation of Absolute Error')
    axs[2, 0].grid(True)

    # Bottom middle subplot: standard deviation plot (log-transformed)
    axs[2, 1].bar(angles, log_std_devs, color='skyblue', alpha=0.7)
    axs[2, 1].set_title('Standard Deviation of Log Transformed Absolute Error vs. Tilt Angle')
    axs[2, 1].set_xlabel('Tilt Angle (degrees)')
    axs[2, 1].set_ylabel('Standard Deviation of Log Transformed Absolute Error')
    axs[2, 1].grid(True)

    # Bottom right subplot: standard deviation plot (sqrt-transformed)
    axs[2, 2].bar(angles, sqrt_std_devs, color='skyblue', alpha=0.7)
    axs[2, 2].set_title('Standard Deviation of Sqrt Transformed Absolute Error vs. Tilt Angle')
    axs[2, 2].set_xlabel('Tilt Angle (degrees)')
    axs[2, 2].set_ylabel('Standard Deviation of Sqrt Transformed Absolute Error')
    axs[2, 2].grid(True)

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
