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

def create_spectrogram(file_path, output_folder, with_labels, range_bins):
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
            timestamps = hdf5_file['data/timestamps'][:]
            timestamps = timestamps / 1e9  # Assuming timestamps are in nanoseconds
            
            # Convert to a NumPy array
            data_array = np.array(data)
            
            # If the data is 3D, we need to select a slice to generate a 2D spectrogram
            if len(data_array.shape) == 3:
                data_array = data_array[0, :, :]  # Using the first frame for the spectrogram
            elif len(data_array.shape) != 2:
                print(f"Unexpected data shape {data_array.shape} in {file_path}. Skipping...")
                return

            # Get the base name of the file (without directory and extension)
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Parse the filename for details
            details = parse_filename(file_path)

            # Parameters for the spectrogram
            NFFT = 256  # Number of data points used in each block for the FFT
            pad_to = 1024
            sampling_period = np.mean(np.diff(timestamps[0,:]))
            sampling_freq =  1 / sampling_period
            noverlap = int(NFFT * 3/4)  # Overlap between segments

            for range_bin in range_bins:
                if range_bin < 0 or range_bin >= data_array.shape[0]:
                    print(f"Range bin {range_bin} is out of bounds for file {file_path}. Skipping this range bin.")
                    continue

                data_array_transposed = data_array[range_bin, :]

                plt.figure(figsize=(10, 6))
                Pxx, freqs, bins, im = plt.specgram(data_array_transposed, NFFT=NFFT, Fs=sampling_freq, noverlap=noverlap)

                plt.colorbar(label='Intensity')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.ylim(0, 1500) # Limit to show only signals from 0 to 1500
                # plt.xlim(0, 0.25) # Limit to show only signals from 0 to 0.25 seconds

                if with_labels:
                    plt.title(f'Spectrogram of {base_name} - Range Bin {range_bin}')
                    # Label propeller
                    propeller_mapping = {
                        'fr': 'front_right',
                        'br': 'back_right',
                        'fl': 'front_left',
                        'bl': 'back_left'
                    }
                    propeller = propeller_mapping.get(details['propeller'], '')
                    if propeller:
                        # Find ground truth frequency
                        hdf5_path = f'parameters/prop_frequency/{propeller}/avg'
                        with h5py.File(file_path, 'r') as hdf5_file:
                            exp_freq = hdf5_file[hdf5_path][:]
                            exp_freq_first = round(exp_freq[0])

                        if details:
                            text_str = (f"Drone Name: {details['drone_name']}\n"
                                        f"Time Stamp: {details['time_stamp']}\n"
                                        f"Tilt Angle: {details['tilt_angle']} degrees\n"
                                        f"Propeller: {propeller}\n"
                                        f"Throttle: {details['throttle']}\n"
                                        f"Expected Frequency: {exp_freq_first}")
                            plt.gcf().text(0.98, 0.95, text_str, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
                else:
                    plt.axis('off')  # Turn off axes

                # Save the spectrogram to an image file
                output_image_path = os.path.join(output_folder, f"{base_name}.png")
                if with_labels:
                    plt.savefig(output_image_path)
                else:
                    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)

                plt.close()

                print(f"Spectrogram for range bin {range_bin} saved to {output_image_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate spectrograms from .mat or HDF5 files in a specified folder.")
    parser.add_argument('input_folder', type=str, help="Path to the folder containing .mat or HDF5 files.")
    parser.add_argument('--output_folder', type=str, default=None, help="Path to the output folder where spectrograms will be saved. Default is './spectrograms'.")
    parser.add_argument('--with_labels', action='store_true', help="Include labels in the spectrograms.")
    parser.add_argument('--range_bins', type=str, default="0", help="Specify a single range bin or a range of range bins (e.g., 120 or 120-130).")

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

    # Parse range bins
    if '-' in args.range_bins:
        start, end = map(int, args.range_bins.split('-'))
        range_bins = list(range(start, end + 1))
    else:
        range_bins = [int(args.range_bins)]

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if filename.endswith('.mat') or filename.endswith('.h5') or filename.endswith('.hdf5'):
            create_spectrogram(file_path, output_folder, args.with_labels, range_bins)

if __name__ == "__main__":
    main()
