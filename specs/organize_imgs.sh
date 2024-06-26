#!/bin/bash

# Base directory where the split_specs are stored
BASE_DIR="split_specs"

# Directory where the raw images will be stored
DEST_DIR="raw_splits"

# Subdirectories to process
SUBDIRS=("train" "test" "val")

# Loop through each subdirectory (train, test, val)
for SUBDIR in "${SUBDIRS[@]}"; do
  # Path to the destination raw directory within the DEST_DIR
  RAW_DIR="${DEST_DIR}/${SUBDIR}"

  # Create the destination raw directory if it doesn't exist
  mkdir -p "${RAW_DIR}"

  # Find and copy all raw images into the destination raw directory
  find "${BASE_DIR}/${SUBDIR}" -type f -path "*/Raw/*.png" -exec cp {} "${RAW_DIR}" \;
  
  echo "Raw images have been copied to ${RAW_DIR}."
done
