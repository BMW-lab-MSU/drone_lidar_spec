import scipy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def create_spectrogram(file_path, output_folder, with_labels):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.mat':
        # Load the .mat file
        mat_file = scipy.io.loadmat(file_path)

        # Extract the 'full_data' key
        full_data = mat_file.get('full_data')
        if full_data is None:
            print(f"No 'full_data' key found in {file_path}. Skipping...")
            return

        # Extract the main data array
        data_array = full_data[0, 0][0]

    elif file_extension in ['.h5', '.hdf5']:
        # Load the HDF5 file
        with h5py.File(file_path, 'r') as hdf5_file:
            # Extract the 'full_data' dataset
            if 'full_data' not in hdf5_file:
                print(f"No 'full_data' dataset found in {file_path}. Skipping...")
                return
            
            full_data = hdf5_file['full_data']
            # Assuming 'full_data' is a 2D array, if it's 3D or other, modify accordingly
            data_array = np.array(full_data)

    else:
        print(f"Unsupported file extension {file_extension}. Skipping {file_path}...")
        return

    # Get the base name of the file (without directory and extension)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Generate the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(data_array, aspect='auto', origin='lower', cmap='viridis')

    if with_labels:
        plt.colorbar(label='Intensity')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title(f'Spectrogram of {base_name}')
    else:
        plt.axis('off')  # Turn off axes

    # Save the spectrogram to an image file
    output_image_path = os.path.join(output_folder, base_name + '_spectrogram.png')
    if with_labels:
        plt.savefig(output_image_path)
    else:
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    
    plt.close()

    print(f"Spectrogram saved to {output_image_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate spectrograms from .mat or HDF5 files in a specified folder.")
    parser.add_argument('input_folder', type=str, help="Path to the folder containing .mat or HDF5 files.")
    parser.add_argument('--output_folder', type=str, default=None, help="Path to the output folder where spectrograms will be saved. Default is './spectrograms'.")
    parser.add_argument('--with_labels', action='store_true', help="Include labels in the spectrograms.")

    args = parser.parse_args()

    input_folder = args.input_folder
    
    # Extract the last part of the input folder path
    last_part_of_input_folder = os.path.basename(os.path.normpath(input_folder))

    # Determine the output subfolder based on the with_labels flag
    subfolder = 'labeled' if args.with_labels else 'raw'
    
    # Construct the output folder path
    output_folder = args.output_folder or os.path.join('.', 'spectrograms', subfolder, last_part_of_input_folder)

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if filename.endswith('.mat') or filename.endswith('.h5') or filename.endswith('.hdf5'):
            create_spectrogram(file_path, output_folder, args.with_labels)

if __name__ == "__main__":
    main()
