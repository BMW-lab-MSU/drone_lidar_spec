import json
import matplotlib.pyplot as plt
import numpy as np
import sys

"""
plot_fill_factors.py

This script reads four JSON files containing metrics data and generates a 2x2 grid of subplots 
showing angle vs MAE with error bars for each JSON file.

Usage:
    python plot_fill_factors.py <json_file1> <json_file2> <json_file3> <json_file4> <output_file>

Arguments:
    <json_file1>  : Path to the first JSON file containing the metrics data.
    <json_file2>  : Path to the second JSON file containing the metrics data.
    <json_file3>  : Path to the third JSON file containing the metrics data.
    <json_file4>  : Path to the fourth JSON file containing the metrics data.
    <output_file> : Path to save the generated plot.

Example:
    python plot_fill_factors.py metrics_data1.json metrics_data2.json metrics_data3.json metrics_data4.json metrics_vs_angle.png

The script will generate a plot with a 2x2 grid of subplots and save it as specified by <output_file>.

JSON File Format:
{
    "angle_1": {
        "MAE": value,
        "SD_MAE": value,
        "MSE": value,
        "SD_MSE": value,
        "RMSE": value,
        "SD_RMSE": value,
        "RMSPE": value,
        "SD_RMSPE": value,
        "Percent Error": value,
        "SD_Percent Error": value,
        "Number of Samples": value
    },
    ...
}
"""

def plot_metrics(json_files, output_file):
    labels = ["Prop-Only Fill Factor", "Partial Fill Factor", "Full Fill Factor", "All Fill Factors"]
    positions = [3, 2, 1, 0]  # Corresponding positions in the input files
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Predicted Frequency MAE vs Angle for All Fill Factors')

    for plot_idx, data_idx in enumerate(positions):
        json_file = json_files[data_idx]
        label = labels[plot_idx]

        with open(json_file, 'r') as f:
            data = json.load(f)

        angles = sorted(data.keys(), key=lambda x: int(x))
        mae = []
        sd_mae = []

        for angle in angles:
            mae_value = data[angle]["MAE"]
            sd_mae_value = data[angle]["SD_MAE"]
            mae.append(mae_value)
            sd_mae.append(sd_mae_value)

        mae = np.array(mae)
        sd_mae = np.array(sd_mae)
        mae_lower = np.maximum(0, mae - sd_mae)  # Truncate at y = 0
        mae_upper = mae + sd_mae

        row, col = divmod(plot_idx, 2)
        ax = axs[row, col]
        ax.errorbar(angles, mae, yerr=[mae - mae_lower, mae_upper - mae], fmt='-o', capsize=5, ecolor='red', color='blue')
        ax.fill_between(angles, mae_lower, mae_upper, color='red', alpha=0.2)
        ax.set_ylim(bottom=0)  # Ensure y = 0 is the bottom line of the plot
        ax.set_title(f'{label}: MAE vs Angle')
        ax.set_xlabel('Angle')
        ax.set_ylabel('MAE')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python plot_fill_factors.py <json_file1> <json_file2> <json_file3> <json_file4> <output_file>")
        sys.exit(1)
    
    json_files = sys.argv[1:5]
    output_file = sys.argv[5]
    plot_metrics(json_files, output_file)
