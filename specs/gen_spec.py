import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py
import argparse
import os

def parse_filename(filename):
    """Extract details from the filename."""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split('-')
    if len(parts) != 9:
        return None
    
    details = {
        'drone_name': f"{parts[0]}-{parts[1]}",
        'time_stamp': f"{parts[2]}-{parts[3]}-{parts[4]}",
        'tilt_angle': parts[6].replace('tilt', ''),
        'propeller': parts[7],
        'throttle': parts[8]
    }
    return details

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
            # Extract the 'data' dataset
            if 'data' not in hdf5_file:
                print(f"No 'data' group found in {file_path}. Skipping...")
                return

            data_group = hdf5_file['data']
            if 'data' not in data_group:
                print(f"No 'data/data' dataset found in {file_path}. Skipping...")
                return
            
            data = data_group['data'][:]
            
            # Convert to a NumPy array
            data_array = np.array(data)
            
            # If the data is 3D, we need to select a slice to generate a 2D spectrogram
            if len(data_array.shape) == 3:
                data_array = data_array[0, :, :]  # Using the first frame for the spectrogram
            elif len(data_array.shape) != 2:
                print(f"Unexpected data shape {data_array.shape} in {file_path}. Skipping...")
                return

    else:
        print(f"Unsupported file extension {file_extension}. Skipping {file_path}...")
        return

    # Get the base name of the file (without directory and extension)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Parse the filename for details
    details = parse_filename(file_path)

    # Parameters for the spectrogram
    NFFT = 256  # Number of data points used in each block for the FFT
    Fs = 1e5    # Sampling frequency
    noverlap = 128  # Overlap between segments

    # Transpose the data array to flip the axes
    data_array_transposed = data_array.T

    # Generate the spectrogram with transposed data
    plt.figure(figsize=(10, 6))
    Pxx, freqs, bins, im = plt.specgram(data_array_transposed.flatten(), NFFT=NFFT, Fs=Fs, noverlap=noverlap, cmap='viridis')

    plt.colorbar(label='Intensity')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    #plt.ylim(0,1500) #Limit frequency range to 0-1500 Hz
    
    if with_labels:
        plt.title(f'Spectrogram of {base_name}')
        if details:
            text_str = (f"Drone Name: {details['drone_name']}\n"
                        f"Time Stamp: {details['time_stamp']}\n"
                        f"Tilt Angle: {details['tilt_angle']} degrees\n"
                        f"Propeller: {details['propeller']}\n"
                        f"Throttle: {details['throttle']}")
            plt.gcf().text(0.98, 0.95, text_str, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
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
