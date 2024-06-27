# Spectrogram Generator 📊🎶

This project generates spectrograms from MATLAB `.mat` files or HDF5 `.h5` files. The input files are stored in a directory named `input_files`, and the generated spectrograms are saved in a directory named `spectrograms`.

## Contents

- [How to Use](#how-to-use)
  - [Running the Script](#running-the-script)
  - [Directory Structure](#directory-structure)
- [Annotations](#annotations)
- [Details File](#details-file)

---

## How to Use 🚀

You can generate spectrograms using a single script.

### Running the Script

1. **Place your `.mat` or `.h5` files in the `input_files` directory.**
2. **Run the script to generate spectrograms, specifying the input folder, output folder, range bins, and number of pixels for bounding boxes.**

#### Command

To generate spectrograms:

```bash
python gen_spec.py /path/to/input/folder --output_folder /path/to/output/folder --range_bins 0-10 --n_pixels 40 --filter_order 1
```

### Arguments

- `/path/to/input/folder` (required): The path to the directory containing your `.mat` or `.h5` files.
- `--output_folder` (optional): The path to the directory where the spectrograms will be saved. If not specified, the script will save the spectrograms in the `spectrograms` folder.
- `--range_bins` (required): Specifies the range of bins to process. You can specify a single bin (e.g., `120`) or a range of bins (e.g., `0-10`).
- `--n_pixels` (optional): The number of pixels around the ground truth frequency for the bounding box. The default value is `40`.
- `--filter_order` (optional): The order of the high pass filter, if applying a filter. 


## Directory Structure 📁

The generated spectrograms will be saved in the following directory structure:

```bash
spectrograms/
├── annotations.json
└── drone_name-timestamp-tiltangle-propeller-throttle
    ├── details.txt
    ├── Labeled
    │   ├── spec_labeled_range_bin=x.png
    │   ├── spec_labeled_range_bin=x+1.png
    │   └── ... (other range bins)
    └── Raw
        ├── spec_raw_range_bin=x.png
        ├── spec_raw_range_bin=x+1.png
        └── ... (other range bins)
```

## Organization

To move the raw images into training, testing, and validation splits for model training, run

```bash
./organize_imgs.sh
```

## Annotations 📝

The `annotations.json` file contains the bounding box annotations for all the processed images in COCO format. It is saved in the `spectrograms` folder and includes the following information for each annotated image:

- `id`: The unique identifier for the annotation.
- `image_id`: The ID of the image.
- `category_id`: The category ID (1 for drone frequency).
- `bbox`: The bounding box coordinates `[x, y, width, height]`.
- `area`: The area of the bounding box.
- `iscrowd`: Specifies whether the annotation is a crowd (always 0 for this case).

## Details File 📄

The `details.txt` file in each subdirectory contains metadata about the spectrograms, including:

- `Drone Name`
- `Time Stamp`
- `Tilt Angle`
- `Propeller`
- `Throttle`
- `Range Bins`
- `Expected Frequency`
- `Fill Factor`

## Enjoy generating your spectrograms! 🎉📈

