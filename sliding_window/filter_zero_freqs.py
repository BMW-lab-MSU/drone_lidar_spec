import json
import sys

"""
filter_zero_prop_frequencies.py

This script reads a JSON file containing data entries, removes entries where all 
prop_frequencies are 0, and writes the filtered data to a new JSON file.

Usage:
    python filter_zero_prop_frequencies.py <input_json_file> <output_json_file>

Arguments:
    <input_json_file>  : Path to the input JSON file containing the data entries.
    <output_json_file> : Path to save the filtered data.

Example:
    python filter_zero_prop_frequencies.py data.json filtered_data.json
"""

def filter_zero_frequencies(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    filtered_data = [entry for entry in data if any(freq != 0 for freq in entry['prop_frequencies'])]

    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_zero_prop_frequencies.py <input_json_file> <output_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    filter_zero_frequencies(input_file, output_file)
