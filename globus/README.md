# 📂 Directory Transfer Script Using Globus CLI

This repository contains scripts to transfer entire directories using the Globus CLI. Specifically, the `load.sh` script is used to load HDF5 (.h5) files from the `blackmore` endpoint to the `tempest` endpoint, and the `save.sh` script is used to save spectrogram files back to the `blackmore` endpoint.

## Prerequisites

**Set up Globus**: To set up Globus, follow [these instructions](https://github.com/BMW-lab-MSU/docs/blob/main/tempest-blackmore-transfer-globus.md) 📖

## Usage

### Running from a Container 🐳

1. **Build or Pull the Container**: Ensure you have a container environment with the Globus CLI installed. You can use a Dockerfile to create this environment or pull from an existing image.

2. **Save the Scripts**: Save the `load.sh` and `save.sh` scripts provided below to your container.

3. **Make the Scripts Executable**: Within the container, run the following commands:
   ```bash
   chmod +x load.sh
   chmod +x save.sh
   ```

4. **Run the Scripts**: Execute the scripts with:
   ```bash
    ./load.sh
    ./save.sh
    ```

### Modifying the Scripts ✏️
Update the `SOURCE_ENDPOINT_ID`, `DESTINATION_ENDPOINT_ID`, `SOURCE_DIRECTORY_PATH`, and `DESTINATION_DIRECTORY_PATH` variables in the `load.sh` and `save.sh` scripts to match your specific endpoints and directory paths.

### Notes 📝
- The `load.sh` script will automatically check if the Globus CLI is installed and if the user is authenticated.
- The `load.sh` script initiates a recursive transfer of the specified directory, includes a file name cleanup step to ensure consistent naming, and monitors the transfer status until completion.
- The `save.sh` script transfers files back to the `blackmore` endpoint without additional file name cleanup.
