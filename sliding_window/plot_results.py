import json
import matplotlib.pyplot as plt
import sys
import numpy as np

"""
plot_results.py

This script reads a JSON file containing metrics data and generates a plot with six subplots showing
angle vs MAE, MSE, RMSE, RMSPE, Percent Error, and SD_MAE with error bars.

Usage:
    python plot_results.py <json_file> <output_file>

Arguments:
    <json_file>  : Path to the JSON file containing the metrics data.
    <output_file>: Path to save the generated plot.

Example:
    python plot_results.py metrics_data.json metrics_vs_angle.png

The script will generate a plot with six subplots and save it as specified by <output_file>.

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

def plot_metrics(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    angles = sorted(data.keys(), key=lambda x: int(x))
    metrics = {
        "MAE": [],
        "SD_MAE": [],
        "MSE": [],
        "SD_MSE": [],
        "RMSE": [],
        "SD_RMSE": [],
        "RMSPE": [],
        "SD_RMSPE": [],
        "Percent Error": [],
        "SD_Percent Error": []
    }

    for angle in angles:
        metrics["MAE"].append(data[angle]["MAE"])
        metrics["SD_MAE"].append(data[angle]["SD_MAE"])
        metrics["MSE"].append(data[angle]["MSE"])
        metrics["SD_MSE"].append(data[angle]["SD_MSE"])
        metrics["RMSE"].append(data[angle]["RMSE"])
        metrics["SD_RMSE"].append(data[angle]["SD_RMSE"])
        metrics["RMSPE"].append(data[angle]["RMSPE"])
        metrics["SD_RMSPE"].append(data[angle]["SD_RMSPE"])
        metrics["Percent Error"].append(data[angle]["Percent Error"])
        metrics["SD_Percent Error"].append(data[angle]["SD_Percent Error"])

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Metrics vs Angle')

    def plot_with_error_bars(ax, angles, values, errors, title, ylabel):
        ax.errorbar(angles, values, yerr=errors, fmt='-o', capsize=5, ecolor='red', color='blue')
        ax.fill_between(angles, np.array(values) - np.array(errors), np.array(values) + np.array(errors), color='red', alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel('Angle')
        ax.set_ylabel(ylabel)

    plot_with_error_bars(axs[0, 0], angles, metrics["MAE"], metrics["SD_MAE"], 'MAE vs Angle', 'MAE')
    plot_with_error_bars(axs[0, 1], angles, metrics["MSE"], metrics["SD_MSE"], 'MSE vs Angle', 'MSE')
    plot_with_error_bars(axs[1, 0], angles, metrics["RMSE"], metrics["SD_RMSE"], 'RMSE vs Angle', 'RMSE')
    plot_with_error_bars(axs[1, 1], angles, metrics["RMSPE"], metrics["SD_RMSPE"], 'RMSPE vs Angle', 'RMSPE')
    plot_with_error_bars(axs[2, 0], angles, metrics["Percent Error"], metrics["SD_Percent Error"], 'Percent Error vs Angle', 'Percent Error')
    plot_with_error_bars(axs[2, 1], angles, metrics["SD_MAE"], [0]*len(metrics["SD_MAE"]), 'SD_MAE vs Angle', 'SD_MAE')  # No error bars for SD_MAE itself

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_results.py <json_file> <output_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_file = sys.argv[2]
    plot_metrics(json_file, output_file)
