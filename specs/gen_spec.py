import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py
import os
import json
import argparse

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

def create_spectrogram(file_path, output_folder, range_bins, n_pixels, coco_output, details):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.mat':
        mat_file = scipy.io.loadmat(file_path)
        full_data = mat_file.get('full_data')
        if full_data is None:
            print(f"No 'full_data' key found in {file_path}. Skipping...")
            return
        data_array = full_data[0, 0][0]
    elif file_extension in ['.h5', '.hdf5']:
        with h5py.File(file_path, 'r') as hdf5_file:
            if 'data' not in hdf5_file:
                print(f"No 'data' group found in {file_path}. Skipping...")
                return
            data_group = hdf5_file['data']
            if 'data' not in data_group:
                print(f"No 'data/data' dataset found in {file_path}. Skipping...")
                return
            data = data_group['data'][:]
            timestamps = hdf5_file['data/timestamps'][:]
            timestamps = timestamps / 1e9
            data_array = np.array(data)
            if len(data_array.shape) == 3:
                data_array = data_array[0, :, :]
            elif len(data_array.shape) != 2:
                print(f"Unexpected data shape {data_array.shape} in {file_path}. Skipping...")
                return
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            NFFT = 256
            pad_to = 1024
            sampling_period = np.mean(np.diff(timestamps[0,:]))
            sampling_freq =  1 / sampling_period
            noverlap = int(NFFT * 3/4)
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
                plt.ylim(0, 1500)
                propeller_mapping = {
                    'fr': 'front_right',
                    'br': 'back_right',
                    'fl': 'front_left',
                    'bl': 'back_left'
                }
                propeller = propeller_mapping.get(details['propeller'], '')
                if propeller:
                    hdf5_path = f'parameters/prop_frequency/{propeller}/avg'
                    with h5py.File(file_path, 'r') as hdf5_file:
                        exp_freq = hdf5_file[hdf5_path][:]
                        exp_freq_first = round(exp_freq[0])
                    freq_index = np.abs(freqs - exp_freq_first).argmin()
                    bbox_y = freqs[freq_index] - n_pixels
                    bbox_height = 2 * n_pixels
                    bbox = [0, int(bbox_y), len(bins), int(bbox_height)]
                    annotation = {
                        "image_id": base_name,
                        "category_id": 1,
                        "bbox": [int(coord) for coord in bbox],
                        "area": int(bbox[2] * bbox[3]),
                        "iscrowd": 0
                    }
                    coco_output["annotations"].append(annotation)
                    
                    # Plot the labeled spectrogram with bounding box in orange
                    plt.axhline(y=exp_freq_first, color='r', linestyle='--')
                    plt.gca().add_patch(plt.Rectangle((0, bbox_y), len(bins), bbox_height, linewidth=1, edgecolor='orange', facecolor='none'))
                    text_str = (f"Drone Name: {details['drone_name']}\n"
                                f"Time Stamp: {details['time_stamp']}\n"
                                f"Tilt Angle: {details['tilt_angle']} degrees\n"
                                f"Propeller: {propeller}\n"
                                f"Throttle: {details['throttle']}\n"
                                f"Expected Frequency: {exp_freq_first}")
                    plt.gcf().text(0.98, 0.95, text_str, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
                    details['expected_frequency'] = int(exp_freq_first)
                output_image_path_labeled = os.path.join(output_folder, 'Labeled', f"spec_labeled_range_bin={range_bin}.png")
                plt.savefig(output_image_path_labeled)
                plt.close()
                
                # Plot the raw spectrogram without bounding box
                plt.figure(figsize=(10, 6))
                plt.specgram(data_array_transposed, NFFT=NFFT, Fs=sampling_freq, noverlap=noverlap)
                plt.axis('off')
                plt.gca().xaxis.set_visible(False)
                plt.gca().yaxis.set_visible(False)
                plt.gca().set_frame_on(False)
                output_image_path_raw = os.path.join(output_folder, 'Raw', f"spec_raw_range_bin={range_bin}.png")
                plt.savefig(output_image_path_raw, bbox_inches='tight', pad_inches=0)
                plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate spectrograms from .mat or HDF5 files in a specified folder.")
    parser.add_argument('input_folder', type=str, help="Path to the folder containing .mat or HDF5 files.")
    parser.add_argument('--n_pixels', type=int, default=30, help="Number of pixels around the ground truth frequency for the bounding box.")
    parser.add_argument('--range_bins', type=str, default="0", help="Specify a single range bin or a range of range bins (e.g., 120 or 120-130).")
    parser.add_argument('--output_folder', type=str, default=None, help="Path to the output folder where spectrograms will be saved. Default is './spectrograms'.")

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder or './spectrograms'
    os.makedirs(output_folder, exist_ok=True)
    if '-' in args.range_bins:
        start, end = map(int, args.range_bins.split('-'))
        range_bins = list(range(start, end + 1))
    else:
        range_bins = [int(args.range_bins)]

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if filename.endswith('.mat') or filename.endswith('.h5') or filename.endswith('.hdf5'):
            details = parse_filename(filename)
            if not details:
                continue
            drone_folder = f"{details['drone_name']}-{details['time_stamp']}-{details['tilt_angle']}-{details['propeller']}-{details['throttle']}"
            drone_output_folder = os.path.join(output_folder, drone_folder)
            labeled_folder = os.path.join(drone_output_folder, 'Labeled')
            raw_folder = os.path.join(drone_output_folder, 'Raw')
            os.makedirs(labeled_folder, exist_ok=True)
            os.makedirs(raw_folder, exist_ok=True)

            # Write details.txt
            details_file_path = os.path.join(drone_output_folder, 'details.txt')
            with open(details_file_path, 'w') as details_file:
                for key, value in details.items():
                    details_file.write(f"{key}: {value}\n")
                details_file.write(f"Range Bins: {args.range_bins}\n")

            coco_output = {
                "images": [],
                "annotations": [],
                "categories": [
                    {
                        "id": 1,
                        "name": "drone_frequency",
                        "supercategory": "object"
                    }
                ]
            }

            create_spectrogram(file_path, drone_output_folder, range_bins, args.n_pixels, coco_output, details)

            # Write annotations.json
            coco_file_path = os.path.join(drone_output_folder, 'annotations.json')
            with open(coco_file_path, 'w') as coco_file:
                json.dump(coco_output, coco_file, indent=4)

            # Write updated details.txt with expected frequency
            with open(details_file_path, 'w') as details_file:
                for key, value in details.items():
                    details_file.write(f"{key}: {value}\n")
                details_file.write(f"range bins: {range_bins}")

            print(f"Processed {filename}")

if __name__ == "__main__":
    main()
