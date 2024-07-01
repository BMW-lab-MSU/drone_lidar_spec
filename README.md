# 2024 Zach Crennen MSU ECE Summer REU Project 🎓

## Research Question ❓
What range of tilt angles can frequencies of a drone rotor be identified in spectral analysis of LiDAR pulses on a stationary drone rotor?

## Directories 📂

### 📜 Scripts
This directory contains a collection of miscellaneous scripts that are crucial for specific data processing and model training tasks.

- **calc_norms.py** 🧮
    - Calculates normalization values for consistent input scaling across the dataset.
    
- **split_data.py** 🔀
    - Splits HDF5 files into training, validation, and test sets, ensuring balanced data distribution.

### 📁 configs
🛠️ Stores custom configuration files for mmdet models, facilitating specific adaptations or optimizations needed for the project.

### det_outputs
🖼️ Some outputs from random images from COCO using the mmdet inference demo.
*Note: These will need to be deleted in the final version.*

### mmdetection
📚 The cloned repository of the mmdetection library.
Includes a bit of customization necessary to get the inference demo running.

### slurm
💻 Stores scripts for running larger scale jobs on Tempest.

### specs
📊 Contains code for generating spectrograms. Generates annotations as well. More details can be found in the [specs/README.md](specs/README.md) file.

### globus
🌐 Contains scripts for transferring files using the Globus CLI.

## Docker 🐳
The Docker image is contained here. For use at MSU, the image was converted to an Apptainer image so it could be run on Tempest (MSU HPC).   
*Note: The Apptainer container (.sif) is not included as the file is too large.*

## Attribution

This project includes files from the [mmdetection](https://github.com/open-mmlab/mmdetection) project, which is licensed under the Apache License 2.0. For more information, see the LICENSE-APACHE-2.0 file.
