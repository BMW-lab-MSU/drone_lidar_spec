import cv2
import numpy as np
import json
import os
import argparse
import re

"""
Script to run a sliding window across images and calculate the average RGB values and standard deviations for each window.
Outputs the window with the farthest RGB values from a specified target for each image and predicts the frequency.

Usage:
    python sliding_window_rgb_stats.py --images_path path_to_images_folder --output_path output.json

Arguments:
    --images_path: Path to the folder containing the images.
    --output_path: Path to the output JSON file where the results will be saved.

The sliding window will have a size of 775xWINDOW_HEIGHT with a stride of 1.
"""

TARGET_MEAN_RGB = [146.2918103158602, 205.90643800403225, 72.32910836693549]

FREQ_RANGE = 1952  # Frequency range from 0 to 1952 Hz
IMG_HEIGHT = 462   # Height of the spectrogram image
WINDOW_HEIGHT = 20 # Height of the sliding window

def sliding_window(image, window_size, stride):
    windows = []
    h, w = image.shape[:2]
    win_w, win_h = window_size
    
    for y in range(0, h - win_h + 1, stride):
        window = image[y:y + win_h, :win_w]  # Extract window
        windows.append((0, y, window))
    
    return windows

def calculate_rgb_error(window, target_rgb):
    if window.shape[2] == 4:
        window = window[:, :, :3]
    mean_rgb = np.mean(window, axis=(0, 1))
    color_error = np.sum(np.abs(mean_rgb - target_rgb))
    return mean_rgb, color_error

def predict_frequency(y_coord):
    # Calculate the center y-coordinate of the window
    y_center = y_coord + (WINDOW_HEIGHT / 2)

    # Reverse the scaling to get the frequency
    frequency = (y_center / IMG_HEIGHT) * FREQ_RANGE

    return FREQ_RANGE - frequency  # Adjust the prediction

def extract_tilt_from_filename(filename):
    # Regex to extract tilt angle assuming the format contains "tilt-<angle>"
    match = re.search(r'tilt-(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def extract_base_filename(filename):
    # Extract everything before "_range_bin" in the filename
    base_name = filename.split('_range_bin')[0]
    return base_name

def extract_time_slice_from_filename(filename):
    # Regex to extract time slice assuming the format contains "_time_slice=<number>"
    match = re.search(r'_time_slice=(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def process_images(images_path):
    results = []
    window_size = (775, WINDOW_HEIGHT)
    stride = 1
    
    for file in os.listdir(images_path):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(images_path, file)
            image = cv2.imread(image_path)
            tilt_angle = extract_tilt_from_filename(file)
            base_filename = extract_base_filename(file)
            time_slice = extract_time_slice_from_filename(file)
            
            if image is not None:
                print(f"Processing image: {file}")
                windows = sliding_window(image, window_size, stride)
                
                max_color_error = float('-inf')
                best_window = None
                
                for idx, (x, y, window) in enumerate(windows):
                    mean_rgb, color_error = calculate_rgb_error(window, TARGET_MEAN_RGB)
                    
                    if color_error > max_color_error:
                        max_color_error = color_error
                        predicted_frequency = predict_frequency(y)
                        best_window = {
                            'window_position': (x, y),
                            'window_dimensions': window_size,
                            'mean_rgb': mean_rgb.tolist(),
                            'color_error': color_error,
                            'predicted_frequency': predicted_frequency
                        }
                    
                    if idx % 100 == 0:
                        print(f"Processed {idx} windows for image: {file}")
                
                image_result = {
                    'image': file,
                    'tilt_angle': tilt_angle,
                    'base_filename': base_filename,
                    'time_slice': time_slice,
                    'best_window': best_window
                }
                results.append(image_result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Sliding window RGB stats for images')
    parser.add_argument('--images_path', required=True, help='Path to the folder containing the images')
    parser.add_argument('--output_path', required=True, help='Path to the output JSON file')
    
    args = parser.parse_args()
    
    print(f"Starting processing for images in: {args.images_path}")
    results = process_images(args.images_path)
    
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Sliding window processing completed. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
