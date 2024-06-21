#!/bin/bash

# Variables
SOURCE_ENDPOINT_ID="0dc1297f-9868-4c68-8637-c9b6bd65d3aa"                          # Tempest
DESTINATION_ENDPOINT_ID="5485832e-723e-4b52-8472-0410e90902ad"                     # Blackmore
SOURCE_DIRECTORY_PATH="/home/d86p233/Desktop/BMW-spec/specs/spectrograms"          # Tempest
DESTINATION_DIRECTORY_PATH="/ece-bmw-lab/drone-lidar/summer2024/all_spectrograms/" # Blackmore
TRANSFER_LABEL="Save_spec:tempest->blackmore"

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

# Check for active transfers with the same source and destination
ACTIVE_TRANSFER=$(globus task list --filter-status ACTIVE | grep -E "$SOURCE_ENDPOINT_ID.*$DESTINATION_ENDPOINT_ID")

if [ -n "$ACTIVE_TRANSFER" ]; then
    echo "An active transfer with the same source and destination already exists. Please wait for it to complete or cancel it before starting a new one."
    exit 1
fi

# Check if the destination directory exists
EXISTS=$(globus ls $DESTINATION_ENDPOINT_ID:$DESTINATION_DIRECTORY_PATH 2>/dev/null)

if [ -n "$EXISTS" ]; then
    echo "Destination directory exists. Proceeding with transfer..."
else
    echo "Destination directory does not exist. Creating directory..."
    # Create the directory if it does not exist
    globus mkdir $DESTINATION_ENDPOINT_ID:$DESTINATION_DIRECTORY_PATH
fi

# Initiate the transfer
TRANSFER_RESULT=$(globus transfer $SOURCE_ENDPOINT_ID:$SOURCE_DIRECTORY_PATH $DESTINATION_ENDPOINT_ID:$DESTINATION_DIRECTORY_PATH --recursive --label "$TRANSFER_LABEL")

# Check for errors in transfer initiation
if [[ $? -ne 0 ]]; then
    echo "Failed to initiate transfer."
    exit 1
fi

# Extract the Transfer ID from the result
TRANSFER_ID=$(echo $TRANSFER_RESULT | grep -oP '(?<=Task ID: )\S+')

if [ -z "$TRANSFER_ID" ]; then
    echo "Failed to extract Transfer ID. Here is the transfer result for debugging:"
    echo "$TRANSFER_RESULT"
    exit 1
fi

echo "Transfer initiated successfully. Transfer ID: $TRANSFER_ID"

# Monitor the transfer status
START_TIME=$(date +%s)

while true
do
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    ELAPSED_TIME_FORMATTED=$(printf '%02dh:%02dm:%02ds\n' $((ELAPSED_TIME/3600)) $((ELAPSED_TIME%3600/60)) $((ELAPSED_TIME%60)))

    STATUS=$(globus task show $TRANSFER_ID | grep -i "Status:" | awk '{print $2}')
    echo "Current status: $STATUS"
    
    # Extract transfer details
    FILES_SUCCEEDED=$(globus task show $TRANSFER_ID | grep -i "Subtasks Succeeded:" | sed 's/[^0-9]*//g')
    FILES_FAILED=$(globus task show $TRANSFER_ID | grep -i "Subtasks Failed:" | sed 's/[^0-9]*//g')
    FILES_PENDING=$(globus task show $TRANSFER_ID | grep -i "Subtasks Pending:" | sed 's/[^0-9]*//g')
    
    if [ -z "$FILES_SUCCEEDED" ]; then FILES_SUCCEEDED=0; fi
    if [ -z "$FILES_FAILED" ]; then FILES_FAILED=0; fi
    if [ -z "$FILES_PENDING" ]; then FILES_PENDING=0; fi

    FILES_TOTAL=$((FILES_SUCCEEDED + FILES_FAILED + FILES_PENDING))

    if [ "$FILES_TOTAL" -eq 0 ]; then
        echo "Transfer details not available yet..."
    else
        PERCENTAGE=$(awk "BEGIN {printf \"%.2f\", ($FILES_SUCCEEDED/$FILES_TOTAL)*100}")
        echo "Files transferred: $FILES_SUCCEEDED / $FILES_TOTAL ($PERCENTAGE%)"

        # Calculate estimated total time and remaining time
        if (( $(awk "BEGIN {print ($PERCENTAGE > 0)}") )); then
            ESTIMATED_TOTAL_TIME=$(awk "BEGIN {printf \"%d\", ($ELAPSED_TIME / ($FILES_SUCCEEDED/$FILES_TOTAL))}")
            REMAINING_TIME=$((ESTIMATED_TOTAL_TIME - ELAPSED_TIME))
            REMAINING_TIME_FORMATTED=$(printf '%02dh:%02dm:%02ds\n' $((REMAINING_TIME/3600)) $((REMAINING_TIME%3600/60)) $((REMAINING_TIME%60)))

            ESTIMATED_TOTAL_TIME_FORMATTED=$(printf '%02dh:%02dm:%02ds\n' $((ESTIMATED_TOTAL_TIME/3600)) $((ESTIMATED_TOTAL_TIME%3600/60)) $((ESTIMATED_TOTAL_TIME%60)))

            echo "Estimated total time: $ESTIMATED_TOTAL_TIME_FORMATTED"
            echo "Estimated time remaining: $REMAINING_TIME_FORMATTED"
        fi
    fi

    echo "Elapsed time: $ELAPSED_TIME_FORMATTED"

    if [[ "$STATUS" == "SUCCEEDED" ]]; then
        echo "Transfer completed successfully."
        break
    elif [[ "$STATUS" == "FAILED" ]]; then
        echo "Transfer failed."
        break
    else
        echo "Transfer in progress... Checking again in 60 seconds."
        echo "-----------------------------------------------------"
        sleep 60
    fi
done


