# Spectrogram Image Normalization Calculator 📊🎶

This script calculates the mean and standard deviation for each color channel (Red, Green, Blue) of spectrogram images in a given directory. These values are essential for normalizing your images when performing transfer learning with models like those in mmdetection.

## Purpose 🎯

Normalizing images is a crucial preprocessing step in many machine learning workflows. This script helps you compute the normalization values specific to your spectrogram dataset, ensuring that your data is properly scaled for training neural networks.

## How to Use 🛠️

1. **Download the Script:**
   Save the script as `calculate_mean_std.py`.

2. **Run the Script:**
   Open a terminal and execute the script with the directory of your spectrogram images as an argument:

```bash
python calculate_mean_std.py /path/to/your/spectrogram_images
```
Replace /path/to/your/spectrogram_images with the actual path to your directory.

## Output:
The script will print the mean and standard deviation for each color channel:


```bash
Means: (mean_r, mean_g, mean_b)
Standard Deviations: (std_r, std_g, std_b)
```

## Example 🔍
Assuming you have a directory of spectrogram images at ./spectrograms, you would run:

```bash
python calculate_mean_std.py ./spectrograms
```
And the output will be something like:

```bash
Means: (120.34, 115.67, 123.45)
Standard Deviations: (60.21, 59.78, 61.34)
```

## Notes 📝
- Ensure your images are in color (RGB) format.
- The script reads images in BGR format by default (as per OpenCV convention) and computes the statistics accordingly.
- Depending on the number of images and their size, the script may take a while to run.

# Happy normalizing! 🎉