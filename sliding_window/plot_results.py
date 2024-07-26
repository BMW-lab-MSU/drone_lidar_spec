import json
import matplotlib.pyplot as plt
import sys

"""
plot_results.py

This script reads a JSON file containing metrics data and generates a plot with six subplots showing
angle vs MAE, MSE, RMSE, RMSPE, Percent Error, and Standard Deviation of Absolute Errors.

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
        "MSE": value,
        "RMSE": value,
        "RMSPE": value,
        "Percent Error": value,
        "Standard Deviation of Absolute Errors": value,
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
        "MSE": [],
        "RMSE": [],
        "RMSPE": [],
        "Percent Error": [],
        "Standard Deviation of Absolute Errors": []
    }

    for angle in angles:
        metrics["MAE"].append(data[angle]["MAE"])
        metrics["MSE"].append(data[angle]["MSE"])
        metrics["RMSE"].append(data[angle]["RMSE"])
        metrics["RMSPE"].append(data[angle]["RMSPE"])
        metrics["Percent Error"].append(data[angle]["Percent Error"])
        metrics["Standard Deviation of Absolute Errors"].append(data[angle]["Standard Deviation of Absolute Errors"])

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Metrics vs Angle')

    axs[0, 0].plot(angles, metrics["MAE"], marker='o')
    axs[0, 0].set_title('MAE vs Angle')
    axs[0, 0].set_xlabel('Angle')
    axs[0, 0].set_ylabel('MAE')

    axs[0, 1].plot(angles, metrics["MSE"], marker='o')
    axs[0, 1].set_title('MSE vs Angle')
    axs[0, 1].set_xlabel('Angle')
    axs[0, 1].set_ylabel('MSE')

    axs[1, 0].plot(angles, metrics["RMSE"], marker='o')
    axs[1, 0].set_title('RMSE vs Angle')
    axs[1, 0].set_xlabel('Angle')
    axs[1, 0].set_ylabel('RMSE')

    axs[1, 1].plot(angles, metrics["RMSPE"], marker='o')
    axs[1, 1].set_title('RMSPE vs Angle')
    axs[1, 1].set_xlabel('Angle')
    axs[1, 1].set_ylabel('RMSPE')

    axs[2, 0].plot(angles, metrics["Percent Error"], marker='o')
    axs[2, 0].set_title('Percent Error vs Angle')
    axs[2, 0].set_xlabel('Angle')
    axs[2, 0].set_ylabel('Percent Error')

    axs[2, 1].plot(angles, metrics["Standard Deviation of Absolute Errors"], marker='o')
    axs[2, 1].set_title('Standard Deviation of Absolute Errors vs Angle')
    axs[2, 1].set_xlabel('Angle')
    axs[2, 1].set_ylabel('Standard Deviation of Absolute Errors')

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
