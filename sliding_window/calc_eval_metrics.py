import json
import argparse
import numpy as np

"""
Script to evaluate prediction metrics from a JSON file.

Usage:
    python eval_metrics.py --json_path path_to_json_file

Arguments:
    --json_path: Path to the JSON file containing prediction data.
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

def main():
    parser = argparse.ArgumentParser(description='Evaluate prediction metrics from JSON')
    parser.add_argument('--json_path', required=True, help='Path to the JSON file containing prediction data')

    args = parser.parse_args()

    print(f"Loading prediction data from: {args.json_path}")
    data = load_json(args.json_path)
    
    print("Calculating evaluation metrics...")
    metrics = calculate_metrics(data)
    
    print("Evaluation metrics calculated:")
    for metric, value in metrics.items():
        if metric == 'Percent Error':
            print(f"{metric}: {value}%")
        else:
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
