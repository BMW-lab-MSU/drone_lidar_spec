# 2024 Zach Crennen MSU ECE Summer REU Project 🎓

## Research Question ❓
What range of tilt angles can frequencies of a drone rotor be identified in spectral analysis of LiDAR pulses on a stationary drone rotor?

## Directories 📂

### calc_norms
📐 Calculate normalization values used for detection in backbone preprocessing.

### det_outputs
🖼️ Some outputs from random images from COCO using the mmdet inference demo.
*Note: These will need to be deleted in the final version.*

### mmdetection
📚 The cloned repository of the mmdetection library.
Includes a bit of customization necessary to get the inference demo running.

### slurm
💻 Stores scripts for running larger scale jobs on Tempest.

### specs
📊 Contains code for generating spectrograms.

## Docker 🐳
The Docker image is contained here. For use at MSU, the image was converted to an Apptainer image so it could be run on Tempest (MSU HPC).   
*Note: The Apptainer container (.sif) is not included as the file is too large.*
