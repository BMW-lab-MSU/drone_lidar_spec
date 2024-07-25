import h5py
import os
import json
import argparse
import numpy as np

"""
Script to extract tilt angle and prop frequencies from HDF5 files and output the results to a JSON file.

Usage:
    python extract_h5_data.py --directory_path path_to_h5_files --output_path output.json

Arguments:
    --directory_path: Path to the directory containing the HDF5 files.
    --output_path: Path to the output JSON file where the extracted data will be saved.

Example:
    python extract_h5_data.py --directory_path /path/to/h5_files --output_path output.json

The script processes each HDF5 file in the specified directory, extracts the tilt angle and prop frequencies, and stores the results in a JSON file with the following structure:
[
    {
        "filename": "file1.h5",
        "tilt_angle": 15,
        "prop_frequencies": [693.0, 692.0, ..., 694.0]
    },
    ...
]
"""

def extract_data_from_h5(file_path):
    """Extract tilt angle and prop frequencies from an HDF5 file."""
    with h5py.File(file_path, 'r') as hdf5_file:
        if 'parameters/tilt' in hdf5_file:
            tilt_angle = hdf5_file['parameters/tilt'][()]
            tilt_angle = int(tilt_angle)  # Convert to native Python int
        else:
            tilt_angle = None
        
        if 'parameters/prop_frequency/front_right/avg' in hdf5_file:
            prop_frequencies = hdf5_file['parameters/prop_frequency/front_right/avg'][:]
            prop_frequencies = prop_frequencies.astype(float).tolist()  # Convert to list of native Python floats
        else:
            prop_frequencies = []

    return tilt_angle, prop_frequencies

def process_h5_files(directory_path, output_path):
    """Process all HDF5 files in a directory and save the results to a JSON file."""
    results = []

    for filename in os.listdir(directory_path):
        if filename.endswith(('.h5', '.hdf5')):
            file_path = os.path.join(directory_path, filename)
            tilt_angle, prop_frequencies = extract_data_from_h5(file_path)
            
            result = {
                'filename': filename,
                'tilt_angle': tilt_angle,
                'prop_frequencies': prop_frequencies
            }
            results.append(result)
    
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Extract tilt angle and prop frequencies from HDF5 files')
    parser.add_argument('--directory_path', required=True, help='Path to the directory containing HDF5 files')
    parser.add_argument('--output_path', required=True, help='Path to the output JSON file')
    
    args = parser.parse_args()
    
    process_h5_files(args.directory_path, args.output_path)

if __name__ == "__main__":
    main()
