import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def create_spectrogram(mat_file_path, output_folder):
    # Load the .mat file
    mat_file = scipy.io.loadmat(mat_file_path)

    # Extract the 'full_data' key
    full_data = mat_file.get('full_data')
    if full_data is None:
        print(f"No 'full_data' key found in {mat_file_path}. Skipping...")
        return

    # Extract the main data array
    data_array = full_data[0, 0][0]

    # Get the base name of the file (without directory and extension)
    base_name = os.path.splitext(os.path.basename(mat_file_path))[0]

    # Generate the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(data_array, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title(f'Spectrogram of {base_name}')

    # Save the spectrogram to an image file
    output_image_path = os.path.join(output_folder, base_name + '_spectrogram.png')
    plt.savefig(output_image_path)
    plt.close()

    print(f"Spectrogram saved to {output_image_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate spectrograms from .mat files in a specified folder.")
    parser.add_argument('input_folder', type=str, help="Path to the folder containing .mat files.")
    parser.add_argument('--output_folder', type=str, default=None, help="Path to the output folder where spectrograms will be saved. Default is './spectrograms'.")

    args = parser.parse_args()

    input_folder = args.input_folder
    
    # Extract the last part of the input folder path
    last_part_of_input_folder = os.path.basename(os.path.normpath(input_folder))

    # Construct the output folder path
    output_folder = args.output_folder or os.path.join('.', 'spectrograms', last_part_of_input_folder)

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all .mat files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.mat'):
            mat_file_path = os.path.join(input_folder, filename)
            create_spectrogram(mat_file_path, output_folder)

if __name__ == "__main__":
    main()
