2024 Zach Crennen MSU ECE Summer REU Project

Research Question: What range of tilt angles can frequencies of a drone rotor be identified in spectral analysis of LiDAR pulses on a stationary drone rotor?

Directories:

calc_norms
    calculate normalizaiton values used for detection in backbone preprocessing

det_outputs
    some outputs from random images from COCO using the mmdet inference demo
    will need to be deleted in final version

mmdetection
    the cloned repo of mmdetection library
    includes a little bit of customization necessary to get inference demo running

slurm
    stores scripts for running larger scale jobs on tempest

specs
    contains code for generating spectrograms


Docker:
    docker image is contained here
    for use at MSU, image was converted to apptainer image so it could be run on tempest (MSU HPC)
    apptainer container (.sif) not included as file is too large