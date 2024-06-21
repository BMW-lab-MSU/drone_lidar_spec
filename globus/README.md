# Directory Transfer Script Using Globus CLI

This repository contains a script to transfer entire directories using the Globus CLI. Specifically, the script `load.sh` is used to load HDF5 (.h5) files from the `blackmore` endpoint to the `tempest` endpoint.

## Prerequisites

**Set up Globus**: To set up Globus, follow [these instructions](https://github.com/BMW-lab-MSU/docs/blob/main/tempest-blackmore-transfer-globus.md)


## Usage

### Running from a Container

1. **Build or Pull the Container**: Ensure you have a container environment with the Globus CLI installed. You can use a Dockerfile to create this environment or pull from an existing image.

2. **Save the Script**: Save the `load.sh` script provided below to your container.

3. **Make the Script Executable**: Within the container, run the following command:
   ```bash
   chmod +x load.sh
   ```
4. **Run the Script**: Execute the script with:
    ```bash
    ./load.sh
    ```

### Modifying the Script

Update the `SOURCE_ENDPOINT_ID`, `DESTINATION_ENDPOINT_ID`, `SOURCE_DIRECTORY_PATH`, and `DESTINATION_DIRECTORY_PATH` variables in the `load.sh` script to match your specific endpoints and directory paths.

### Notes

- The script will automatically check if the Globus CLI is installed and if the user is authenticated.
- The script initiates a recursive transfer of the specified directory and monitors the transfer status until completion.
- Feel free to modify the script and the instructions to fit your specific needs.
