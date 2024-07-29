import json
import matplotlib.pyplot as plt
import numpy as np
import sys

"""
plot_total_absolute_error_scatters.py

This script reads four JSON files containing data entries and generates a 2x2 grid of scatter plots 
showing tilt_angle vs total_absolute_error from the best_window for each JSON file. Each plot is labeled according to the filename,
and includes markers for the mean and error bars just to the right side of all the data points.

Usage:
    python plot_total_absolute_error_scatters.py <json_file1> <json_file2> <json_file3> <json_file4> <output_file> [<offset>] [--show-counts]

Arguments:
    <json_file1>  : Path to the first JSON file containing the data entries.
    <json_file2>  : Path to the second JSON file containing the data entries.
    <json_file3>  : Path to the third JSON file containing the data entries.
    <json_file4>  : Path to the fourth JSON file containing the data entries.
    <output_file> : Path to save the generated plot.
    [<offset>]    : Optional. Offset for the error bars. Default is 1.0.
    [--show-counts]: Optional. Include counts of data points for each angle.

Example:
    python plot_total_absolute_error_scatters.py /home/d86p233/Desktop/BMW-spec/work_dirs/SLIDE_noboost_prop-only/noboost_prop-only-comparisons.json /home/d86p233/Desktop/BMW-spec/work_dirs/SLIDE_noboost_partial_SF/noboost_partial_SF-comparisons.json /home/d86p233/Desktop/BMW-spec/work_dirs/SLIDE_noboost_full_SF/noboost_full_SF-comparisons.json /home/d86p233/Desktop/BMW-spec/work_dirs/SLIDE_noboost_all_SF/noboost_all_SF-comparisons.json total_absolute_error_vs_tilt.png 2.0 --show-counts

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
        "best_window": {
            "window_position": [
                0,
                365
            ],
            "window_dimensions": [
                775,
                20
            ],
            "mean_rgb": [
                66.56664516129032,
                209.6343870967742,
                144.00193548387097
            ],
            "std_rgb": [
                30.98150348873404,
                20.628722650051376,
                49.5931706679358
            ],
            "absolute_error": [
                5.031811659946229,
                3.3080073924731153,
                3.638309811827952
            ],
            "total_absolute_error": 11.978128864247296,
            "total_squared_error": 49.49933977667114,
            "squared_error": [
                25.31912858117082,
                10.94291290865678,
                13.237298286843547
            ],
            "predicted_frequency": 367.58441558441564
        }
    },
    ...
]
"""

def plot_total_absolute_error(json_files, output_file, offset=1.0, show_counts=False):
    labels = [
        "Prop-Only Fill Factor", 
        "Partial Fill Factor", 
        "Full Fill Factor", 
        "All Fill Factors"
    ]
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Tilt Angle vs Total Absolute Error Scatterplots for All Fill Factors')

    plot_order = [0, 1, 2, 3]  # Order: top left, top right, bottom left, bottom right
    for idx in plot_order:
        json_file = json_files[idx]
        label = labels[idx]

        with open(json_file, 'r') as f:
            data = json.load(f)

        tilt_angles = [entry['tilt_angle'] for entry in data]
        total_absolute_errors = [entry['best_window']['total_absolute_error'] for entry in data]

        row, col = divmod(plot_order.index(idx), 2)
        ax = axs[row, col]
        scatter = ax.scatter(tilt_angles, total_absolute_errors, label='Total Absolute Error')

        unique_angles = sorted(set(tilt_angles))
        mean_errors = []
        std_errors = []
        num_points = []

        for angle in unique_angles:
            errors_at_angle = [total_absolute_errors[i] for i in range(len(tilt_angles)) if tilt_angles[i] == angle]
            mean_errors.append(np.mean(errors_at_angle))
            std_errors.append(np.std(errors_at_angle))
            num_points.append(len(errors_at_angle))

        # Offset the error bars slightly to the right
        offset_angles = [angle + offset for angle in unique_angles]
        errorbar = ax.errorbar(offset_angles, mean_errors, yerr=std_errors, fmt='o', color='red', capsize=5, label='Mean ± SD')

        ax.set_title(f'{label}: Tilt Angle vs Total Absolute Error')
        ax.set_xlabel('Tilt Angle')
        ax.set_ylabel('Total Absolute Error')
        ax.legend()

        if show_counts:
            # Annotate number of points for each angle at the top
            for i, angle in enumerate(unique_angles):
                ax.annotate(f'n={num_points[i]}', (angle, ax.get_ylim()[1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 6 or len(sys.argv) > 8:
        print("Usage: python plot_total_absolute_error_scatters.py <json_file1> <json_file2> <json_file3> <json_file4> <output_file> [<offset>] [--show-counts]")
        sys.exit(1)
    
    json_files = sys.argv[1:5]
    output_file = sys.argv[5]
    offset = float(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith("--") else 1.0
    show_counts = "--show-counts" in sys.argv
    plot_total_absolute_error(json_files, output_file, offset, show_counts)
