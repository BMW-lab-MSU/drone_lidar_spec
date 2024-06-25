import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py
import os
import json
import argparse
from PIL import Image

def parse_filename(filename):
    """Extract details from the filename."""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split('-')
    details = {
        'drone_name': '-'.join(parts[:len(parts) - 8 + 1]),
        'time_stamp': f"{parts[-7]}-{parts[-6]}-{parts[-5]}",
        'tilt_angle': parts[-3],  # Placeholder, will be updated from the HDF5 file
        'propeller': parts[-2],
        'throttle': parts[-1]
    }
    return details

def read_tilt_angle(file_path):
    """Read the tilt angle from the HDF5 file."""
    with h5py.File(file_path, 'r') as hdf5_file:
        if 'parameters/tilt' in hdf5_file:
            tilt_angle = hdf5_file['parameters/tilt'][()]
            return tilt_angle
    return None

def read_fill_factor(file_path):
    """Read the fill factor from the HDF5 file."""
    with h5py.File(file_path, 'r') as hdf5_file:
        if 'parameters/fill_factor' in hdf5_file:
            fill_factor = hdf5_file['parameters/fill_factor'][()]
            return fill_factor
    return None

def create_spectrogram(file_path, labeled_folder, raw_folder, range_bins, n_pixels, coco_output, details, dimensions, image_id):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.mat':
        mat_file = scipy.io.loadmat(file_path)
        full_data = mat_file.get('full_data')
        if full_data is None:
            print(f"No 'full_data' key found in {file_path}. Skipping...")
            return dimensions, image_id
        data_array = full_data[0, 0][0]
    elif file_extension in ['.h5', '.hdf5']:
        with h5py.File(file_path, 'r') as hdf5_file:
            if 'data' not in hdf5_file:
                print(f"No 'data' group found in {file_path}. Skipping...")
                return dimensions, image_id
            data_group = hdf5_file['data']
            if 'data' not in data_group:
                print(f"No 'data/data' dataset found in {file_path}. Skipping...")
                return dimensions, image_id
            data = data_group['data'][:]
            timestamps = hdf5_file['data/timestamps'][:]
            timestamps = timestamps / 1e9
            data_array = np.array(data)
            if len(data_array.shape) == 3:
                data_array = data_array[0, :, :]
            elif len(data_array.shape) != 2:
                print(f"Unexpected data shape {data_array.shape} in {file_path}. Skipping...")
                return dimensions, image_id
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
                        "id": len(coco_output["annotations"]) + 1,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [int(coord) for coord in bbox],
                        "area": int(bbox[2] * bbox[3]),
                        "iscrowd": 0
                    }
                    coco_output["annotations"].append(annotation)
                    
                    # Plot the labeled spectrogram with bounding box in orange
                    plt.axhline(y=exp_freq_first, color='r', linestyle='--')
                    plt.gca().add_patch(plt.Rectangle((0, bbox_y), len(bins), bbox_height, linewidth=1, edgecolor='orange', facecolor='none'))
                    fill_factor = read_fill_factor(file_path)
                    text_str = (f"Drone Name: {details['drone_name']}\n"
                                f"Time Stamp: {details['time_stamp']}\n"
                                f"Tilt Angle: {details['tilt_angle']} degrees\n"
                                f"Propeller: {propeller}\n"
                                f"Throttle: {details['throttle']}\n"
                                f"Actual Frequency: {exp_freq_first}\n"
                                f"Fill Factor: {fill_factor}\n"
                                f"Range Bin: {range_bin}")
                    plt.gcf().text(0.98, 0.95, text_str, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
                    details['actual_frequency'] = int(exp_freq_first)
                output_image_path_labeled = os.path.join(labeled_folder, f"{base_name}_range_bin={range_bin}.png")
                plt.savefig(output_image_path_labeled)
                plt.close()
                
                # Plot the raw spectrogram without bounding box
                plt.figure(figsize=(10, 6))
                plt.specgram(data_array_transposed, NFFT=NFFT, Fs=sampling_freq, noverlap=noverlap)
                plt.axis('off')
                plt.gca().xaxis.set_visible(False)
                plt.gca().yaxis.set_visible(False)
                plt.gca().set_frame_on(False)
                output_image_path_raw = os.path.join(raw_folder, f"{base_name}_range_bin={range_bin}.png")
                plt.savefig(output_image_path_raw, bbox_inches='tight', pad_inches=0)
                plt.close()
            
                if dimensions is None:
                    img = Image.open(output_image_path_raw)
                    dimensions = img.size
                
                # Add image information to COCO output
                image_info = {
                    "id": image_id,
                    "file_name": os.path.basename(output_image_path_labeled),
                    "height": dimensions[1],
                    "width": dimensions[0]
                }
                coco_output["images"].append(image_info)
                image_id += 1
            
    return dimensions, image_id

def main():
    parser = argparse.ArgumentParser(description="Generate spectrograms from .mat or HDF5 files in a specified folder.")
    parser.add_argument('input_folder', type=str, help="Path to the folder containing .mat or HDF5 files.")
    parser.add_argument('--n_pixels', type=int, default=40, help="Number of pixels around the ground truth frequency for the bounding box.")
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

    all_annotations = {
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
    
    dimensions = None
    image_id = 1

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if filename.endswith('.mat') or filename.endswith('.h5') or filename.endswith('.hdf5'):
            details = parse_filename(filename)
            if not details:
                continue
            
            # Update tilt angle from the HDF5 file
            tilt_angle = read_tilt_angle(file_path)
            if tilt_angle is not None:
                details['tilt_angle'] = int(tilt_angle)
            
            drone_folder = f"{details['drone_name']}-{details['time_stamp']}-{details['tilt_angle']}-{details['propeller']}-{details['throttle']}"
            drone_output_folder = os.path.join(output_folder, drone_folder)
            labeled_folder = os.path.join(drone_output_folder, 'Labeled')
            raw_folder = os.path.join(drone_output_folder, 'Raw')
            os.makedirs(labeled_folder, exist_ok=True)
            os.makedirs(raw_folder, exist_ok=True)

            dimensions, image_id = create_spectrogram(file_path, labeled_folder, raw_folder, range_bins, args.n_pixels, all_annotations, details, dimensions, image_id)

            # Write details.txt
            details_file_path = os.path.join(drone_output_folder, 'details.txt')
            with open(details_file_path, 'w') as details_file:
                for key, value in details.items():
                    if key != 'actual_frequency':
                        details_file.write(f"{key}: {value}\n")
                details_file.write(f"Range Bins: {args.range_bins}\n")
                if 'actual_frequency' in details:
                    details_file.write(f"Actual Frequency: {details['actual_frequency']}\n")

            print(f"Processed {filename}")

    # Write combined annotations.json
    combined_coco_file_path = os.path.join(output_folder, 'annotations.json')
    with open(combined_coco_file_path, 'w') as coco_file:
        json.dump(all_annotations, coco_file, indent=4)

if __name__ == "__main__":
    main()
