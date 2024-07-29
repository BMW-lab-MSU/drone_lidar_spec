import json
import matplotlib.pyplot as plt
import sys

"""
plot_absolute_error.py

This script reads four JSON files containing data entries and generates a 2x2 grid of scatter plots 
showing tilt_angle vs absolute_error for each JSON file.

Usage:
    python plot_absolute_error.py <json_file1> <json_file2> <json_file3> <json_file4> <output_file>

Arguments:
    <json_file1>  : Path to the first JSON file containing the data entries.
    <json_file2>  : Path to the second JSON file containing the data entries.
    <json_file3>  : Path to the third JSON file containing the data entries.
    <json_file4>  : Path to the fourth JSON file containing the data entries.
    <output_file> : Path to save the generated plot.

Example:
    python plot_absolute_error.py data1.json data2.json data3.json data4.json absolute_error_vs_tilt.png

The script will generate a plot with a 2x2 grid of scatter plots and save it as specified by <output_file>.

JSON File Format:
[
    {
        "image": "filename.png",
        "tilt_angle": 20,
        "time_slice": 15,
        "predicted_frequency": 988.6753246753246,
        "ground_truth_frequency": 1152.4897435897435,
        "absolute_error": 163.81441891441887,
        "squared_error": 26835.163844268714,
        "best_window": { ... }
    },
    ...
]
"""

def plot_absolute_error(json_files, output_file):
    labels = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Tilt Angle vs Absolute Error for Multiple Datasets')

    for idx, json_file in enumerate(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)

        tilt_angles = [entry['tilt_angle'] for entry in data]
        absolute_errors = [entry['absolute_error'] for entry in data]

        row, col = divmod(idx, 2)
        ax = axs[row, col]
        ax.scatter(tilt_angles, absolute_errors)
        ax.set_title(f'{labels[idx]}: Tilt Angle vs Absolute Error')
        ax.set_xlabel('Tilt Angle')
        ax.set_ylabel('Absolute Error')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python plot_absolute_error.py <json_file1> <json_file2> <json_file3> <json_file4> <output_file>")
        sys.exit(1)
    
    json_files = sys.argv[1:5]
    output_file = sys.argv[5]
    plot_absolute_error(json_files, output_file)
