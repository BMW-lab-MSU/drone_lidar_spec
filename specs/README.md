# Spectrogram Generator 📊🎶

This project generates spectrograms from MATLAB `.mat` files. The input `.mat` files are stored in a directory named `mat_files`, and the generated spectrograms are saved in a directory named `spectrograms`.

## Contents

- [How to Use](#how-to-use)
  - [Running the Script](#running-the-script)
  - [Directory Structure](#directory-structure)

---

## How to Use 🚀

There are two ways to generate spectrograms: with labels and without labels. This is done by running either `gen_spec_labeled.py` or `gen_spec_raw.py`.


### Running the Script

1. **Place your `.mat` files in the `mat_files` directory.**
2. **Run the appropriate script to generate spectrograms.**

#### With Labels

To generate spectrograms with labels:

```bash
python gen_spec_labeled.py mat_files/FOLDERHERE --output_folder path/to/output/folder
```
#### Without Labels

To generate spectrograms without labels:

```bash
python gen_spec_raw.py mat_files/FOLDERHERE --output_folder path/to/output/folder
```

## Notes

`mat_files` is the input folder containing your `.mat` files.

`spectrograms/labeled` or `spectrograms/raw` are the folders where the spectrograms will be saved.

The `--output_folder` argument is optional, and will automatically save it properly by default

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