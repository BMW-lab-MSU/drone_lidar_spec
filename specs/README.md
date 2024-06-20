# Spectrogram Generator 📊🎶

This project generates spectrograms from MATLAB `.mat` files or HDF5 `.h5` files. The input files are stored in a directory named `input_files`, and the generated spectrograms are saved in a directory named `spectrograms`.

## Contents

- [How to Use](#how-to-use)
  - [Running the Script](#running-the-script)
  - [Directory Structure](#directory-structure)

---

## How to Use 🚀

You can generate spectrograms either with labels or without labels by using a single script and specifying the desired option through a command-line argument.

### Running the Script

1. **Place your `.mat` or `.h5` files in the `mat_files` or `hdf5_files` directory respectively.**
2. **Run the script to generate spectrograms, specifying whether to include labels.**

#### With Labels

To generate spectrograms with labels:

```bash
python gen_spec.py path/to/input/folder --output_folder path/to/output/folder --with_labels
```
#### Without Labels

To generate spectrograms without labels:

```bash
python gen_spec.py path/to/input/folder --output_folder path/to/output/folder
```

## Notes

`spectrograms/labeled` or `spectrograms/raw` are the folders where the spectrograms will be saved.

The `--output_folder` argument is optional. If not specified, the script will automatically save the spectrograms in the appropriate folder based on the presence of labels.

## Directory Structure 📁

The generated spectrograms will be saved in the following directory structure:

```
├── gen_spec_labeled.py
├── gen_spec_raw.py
├── mat_files
│   ├── norm360-drone-prop-distance-test140719
│   └── norm360-drone-prop-distance-test145448
├── README.md
└── spectrograms
    ├── labeled
    │   ├── norm360-drone-prop-distance-test140719
    │   └── norm360-drone-prop-distance-test145448
    └── raw
        ├── norm360-drone-prop-distance-test140719
        └── norm360-drone-prop-distance-test145448
```


# Enjoy generating your spectrograms! 🎉📈

# TODO: UPDATE THIS WITH --range_bins ARG