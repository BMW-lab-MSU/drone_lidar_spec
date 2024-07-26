import json
import argparse
import numpy as np

"""
Script to evaluate prediction metrics from a JSON file and split them by angle.

Usage:
    python eval_metrics.py --json_path path_to_json_file --output_path path_to_output_json_file

Arguments:
    --json_path: Path to the JSON file containing prediction data.
    --output_path: Path to the output JSON file where the evaluation metrics will be saved.
"""

FREQ_RANGE = 1952  # Frequency range from 0 to 1952 Hz

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def calculate_metrics(data):
    absolute_errors = [entry['absolute_error'] for entry in data]
    squared_errors = [entry['squared_error'] for entry in data]
    actual_frequencies = [entry['ground_truth_frequency'] for entry in data]
    predicted_frequencies = [entry['predicted_frequency'] for entry in data]
    
    mae = np.mean(absolute_errors)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    rmspe = np.sqrt(np.mean(((np.array(predicted_frequencies) - np.array(actual_frequencies)) / np.array(actual_frequencies)) ** 2))
    percent_error = (mae / FREQ_RANGE) * 100
    std_abs_error = np.std(absolute_errors)
    num_samples = len(data)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'RMSPE': rmspe,
        'Percent Error': percent_error,
        'Standard Deviation of Absolute Errors': std_abs_error,
        'Number of Samples': num_samples
    }

    return metrics

def group_by_angle(data):
    grouped_data = {}
    for entry in data:
        angle = entry['tilt_angle']
        if angle not in grouped_data:
            grouped_data[angle] = []
        grouped_data[angle].append(entry)
    return grouped_data

def save_json(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Evaluate prediction metrics from JSON')
    parser.add_argument('--json_path', required=True, help='Path to the JSON file containing prediction data')
    parser.add_argument('--output_path', required=True, help='Path to the output JSON file where the evaluation metrics will be saved')

    args = parser.parse_args()

    print(f"Loading prediction data from: {args.json_path}")
    data = load_json(args.json_path)
    
    print("Grouping data by angle...")
    grouped_data = group_by_angle(data)
    
    print("Calculating evaluation metrics...")
    all_metrics = {}
    for angle, entries in grouped_data.items():
        print(f"\nMetrics for tilt angle: {angle}")
        metrics = calculate_metrics(entries)
        all_metrics[angle] = metrics
        
        for metric, value in metrics.items():
            if metric == 'Percent Error':
                print(f"{metric}: {value}%")
            else:
                print(f"{metric}: {value}")
    
    print(f"Saving evaluation metrics to: {args.output_path}")
    save_json(all_metrics, args.output_path)
    print("Evaluation metrics saved.")

if __name__ == "__main__":
    main()
