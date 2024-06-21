#!/bin/bash

# Variables
SOURCE_ENDPOINT_ID="5485832e-723e-4b52-8472-0410e90902ad"                       #Blackmore
DESTINATION_ENDPOINT_ID="0dc1297f-9868-4c68-8637-c9b6bd65d3aa"                  #Tempest
SOURCE_DIRECTORY_PATH="/ece-bmw-lab/drone-lidar/summer2024/all_h5s/"            #Blackmore
DESTINATION_DIRECTORY_PATH="/home/d86p233/Desktop/BMW-spec/specs/hdf5_files"    #Tempest
TRANSFER_LABEL="Load_H5_blackmore->tempest"

# Ensure Globus CLI is installed
if ! command -v globus &> /dev/null
then
    echo "Globus CLI could not be found, please install it first."
    exit 1
fi

# Ensure user is logged into Globus CLI
if ! globus whoami &> /dev/null
then
    echo "Please log in to Globus CLI."
    globus login
fi

# Initiate the transfer
TRANSFER_ID=$(globus transfer $SOURCE_ENDPOINT_ID:$SOURCE_DIRECTORY_PATH $DESTINATION_ENDPOINT_ID:$DESTINATION_DIRECTORY_PATH --recursive --label "$TRANSFER_LABEL" --format=UNIX)

if [ $? -eq 0 ]; then
    echo "Transfer initiated successfully. Transfer ID: $TRANSFER_ID"
else
    echo "Failed to initiate transfer."
    exit 1
fi

# Monitor the transfer status (optional)
while true
do
    STATUS=$(globus task show $TRANSFER_ID --format=UNIX | grep -i "Status:" | awk '{print $2}')
    echo "Current status: $STATUS"

    if [[ "$STATUS" == "SUCCEEDED" ]]; then
        echo "Transfer completed successfully."
        break
    elif [[ "$STATUS" == "FAILED" ]]; then
        echo "Transfer failed."
        break
    else
        echo "Transfer in progress... Checking again in 30 seconds."
        sleep 30
    fi
done
