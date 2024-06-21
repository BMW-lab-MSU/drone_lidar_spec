# 2024 Zach Crennen MSU ECE Summer REU Project 🎓

## Research Question ❓
What range of tilt angles can frequencies of a drone rotor be identified in spectral analysis of LiDAR pulses on a stationary drone rotor?

## Directories 📂

### scripts
Contains miscellaneous infrequently used scripts.

- **calc_norms.py**
    - Calculate the normalization values for backbone training.

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
