import json
import os
import argparse

"""
Script to compare predicted frequencies from image data with ground truth frequencies from H5 data.

Usage:
    python compare_frequencies.py --h5_json path_to_h5_json.json --image_json path_to_image_json.json --output_path output.json

Arguments:
    --h5_json: Path to the JSON file containing H5 data.
    --image_json: Path to the JSON file containing image data.
    --output_path: Path to the output JSON file where the comparison results will be saved.

The script matches the entries based on the base filename and time slice, calculates the absolute error and squared error between
the predicted frequency and the ground truth frequency, and saves the results in the specified output JSON file.
"""

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def match_frequencies(h5_data, image_data):
    results = []

    for image_entry in image_data:
        base_filename = image_entry["base_filename"]
        time_slice = image_entry["time_slice"] - 1  # Adjust to 0-based index
        predicted_frequency = image_entry["best_window"]["predicted_frequency"]

        matching_h5_entry = next((entry for entry in h5_data if entry["filename"].startswith(base_filename)), None)
        if matching_h5_entry:
            ground_truth_frequency = matching_h5_entry["prop_frequencies"][time_slice]
            absolute_error = abs(predicted_frequency - ground_truth_frequency)
            squared_error = (predicted_frequency - ground_truth_frequency) ** 2

            result = {
                "image": image_entry["image"],
                "tilt_angle": image_entry["tilt_angle"],
                "time_slice": time_slice + 1,  # Adjust back to 1-based index for reporting
                "predicted_frequency": predicted_frequency,
                "ground_truth_frequency": ground_truth_frequency,
                "absolute_error": absolute_error,
                "squared_error": squared_error
            }
            results.append(result)

    return results

def save_json(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Compare predicted frequencies with ground truth frequencies')
    parser.add_argument('--h5_json', required=True, help='Path to the JSON file containing H5 data')
    parser.add_argument('--image_json', required=True, help='Path to the JSON file containing image data')
    parser.add_argument('--output_path', required=True, help='Path to the output JSON file')

    args = parser.parse_args()

    h5_data = load_json(args.h5_json)
    image_data = load_json(args.image_json)

    comparison_results = match_frequencies(h5_data, image_data)

    save_json(comparison_results, args.output_path)

    print(f"Comparison completed. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
