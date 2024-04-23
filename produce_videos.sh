#!/bin/bash

# Folder containing the yaml files
FOLDER_PATH="./logs/oracle_oracle/configs/"

# First, modify the yaml files
for file in "$FOLDER_PATH"/*.yaml; do
    # Use sed to perform replacements in-place
    sed -i 's/SAVE_VIDEO_AS_RAW_FRAMES: True/SAVE_VIDEO_AS_RAW_FRAMES: False/g' "$file"
    sed -i 's/VIDEO_INTERVAL: 30/VIDEO_INTERVAL: 2/g' "$file"
done

echo "Modifications to yaml files done!"

# Now, run the command for each yaml file
for file in "$FOLDER_PATH"/*.yaml; do
    # Extract filename without the path and extension
    filename=$(basename -- "$file")
    name="${filename%.*}"

    # Run the command with the extracted name
    ./run_cli.sh oracle_oracle "$name" 2
done

echo "All commands executed!"
