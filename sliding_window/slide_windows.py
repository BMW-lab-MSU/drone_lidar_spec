import cv2
import numpy as np
import json
import os
import argparse
import re

"""
Script to run a sliding window across images and calculate the average RGB values and standard deviations for each window.
Outputs the window with the closest RGB values to a specified target for each image and predicts the frequency.

Usage:
    python sliding_window_rgb_stats.py --images_path path_to_images_folder --output_path output.json

Arguments:
    --images_path: Path to the folder containing the images.
    --output_path: Path to the output JSON file where the results will be saved.

The sliding window will have a size of 775xWINDOW_HEIGHT with a stride of 1.
"""

TARGET_MEAN_RGB = [71.59845682123655, 206.32637970430108, 147.64024529569892]
TARGET_STD_RGB = [41.225951891250425, 25.426265976364576, 81.55837833868854]

FREQ_RANGE = 1952  # Frequency range from 0 to 1952 Hz
IMG_HEIGHT = 462   # Height of the spectrogram image
WINDOW_HEIGHT = 20 # Height of the sliding window

def sliding_window(image, window_size, stride):
    windows = []
    h, w = image.shape[:2]
    win_w, win_h = window_size
    
    for y in range(0, h - win_h + 1, stride):
        for x in range(0, w - win_w + 1, stride):
            window = image[y:y + win_h, x:x + win_w]  # Fix here: Correct slicing for window
            windows.append((x, y, window))
    
    return windows

def calculate_rgb_stats(window):
    mean_rgb = np.mean(window, axis=(0, 1))
    std_rgb = np.std(window, axis=(0, 1))
    return mean_rgb, std_rgb

def calculate_errors(mean_rgb, target_mean_rgb):
    absolute_error = np.abs(mean_rgb - target_mean_rgb)
    squared_error = (mean_rgb - target_mean_rgb) ** 2
    total_absolute_error = np.sum(absolute_error)
    total_squared_error = np.sum(squared_error)
    return total_absolute_error, total_squared_error, absolute_error.tolist(), squared_error.tolist()

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
                
                min_squared_error = float('inf')
                best_window = None
                
                for idx, (x, y, window) in enumerate(windows):
                    mean_rgb, std_rgb = calculate_rgb_stats(window)
                    total_absolute_error, total_squared_error, absolute_error, squared_error = calculate_errors(mean_rgb, TARGET_MEAN_RGB)
                    
                    if total_squared_error < min_squared_error:
                        min_squared_error = total_squared_error
                        predicted_frequency = predict_frequency(y)
                        best_window = {
                            'window_position': (x, y),
                            'window_dimensions': window_size,
                            'mean_rgb': mean_rgb.tolist(),
                            'std_rgb': std_rgb.tolist(),
                            'absolute_error': absolute_error,
                            'total_absolute_error': total_absolute_error,
                            'total_squared_error': total_squared_error,
                            'squared_error': squared_error,
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
