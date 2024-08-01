import json
import argparse
import os

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def extract_sd_mae(data):
    sd_mae_values = []
    for key, value in data.items():
        sd_mae_values.append(value["SD_MAE"])
    return sd_mae_values

def main(json_files):
    all_sd_mae_values = []
    
    for json_file in json_files:
        data = load_json(json_file)
        sd_mae_values = extract_sd_mae(data)
        all_sd_mae_values.extend(sd_mae_values)
    
    average_sd_mae = sum(all_sd_mae_values) / len(all_sd_mae_values)
    print(f"Average SD_MAE: {average_sd_mae}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the average SD_MAE from multiple JSON files.")
    parser.add_argument('--json_files', type=str, nargs=4, required=True, help='Paths to the JSON files.')

    args = parser.parse_args()

    main(args.json_files)
