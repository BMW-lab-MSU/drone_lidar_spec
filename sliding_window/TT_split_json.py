import json
import random
import argparse

"""
This script splits a JSON array into two separate JSON files for training and testing datasets, with an 80/20 split.

Usage:
    python TT_split_json.py --input data.json --train_output train_data.json --test_output test_data.json

Arguments:
    --input        Path to the input JSON file containing the data array.
    --train_output Path to the output JSON file for the training dataset.
    --test_output  Path to the output JSON file for the testing dataset.
"""

def split_json(input_file, train_output_file, test_output_file):
    # Read the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Shuffle the data to ensure randomness
    random.shuffle(data)

    # Determine the split index
    split_index = int(0.8 * len(data))

    # Split the data into training and testing sets
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Write the training data to a JSON file
    with open(train_output_file, 'w') as f:
        json.dump(train_data, f, indent=4)

    # Write the testing data to a JSON file
    with open(test_output_file, 'w') as f:
        json.dump(test_data, f, indent=4)

    print(f"Training data and testing data have been split into '{train_output_file}' and '{test_output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a JSON array into training and testing datasets.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--train_output', type=str, required=True, help='Path to the output JSON file for the training dataset.')
    parser.add_argument('--test_output', type=str, required=True, help='Path to the output JSON file for the testing dataset.')

    args = parser.parse_args()

    split_json(args.input, args.train_output, args.test_output)
