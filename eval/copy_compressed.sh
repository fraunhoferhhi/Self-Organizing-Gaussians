#!/bin/bash
#
# Helper script to package up a set of scenes from a training run into a zip file.
# 
# Copies the compressed files from a training run to a destination directory.
# Only copies one compression (third argument), does not copy any of the training logs or decompressed files.
# Sorts experiments into folders by dataset. The dataset is determined by the first part of the folder name after the first dash and before the first underscore.
# For another directory structure, the dataset name is the string before the first underscore and after the last dash.
# Folders are organized into MipNeRF360, DeepBlending, SyntheticNeRF, and TanksAndTemples based on the dataset prefix.
# After copying, the entire destination directory is zipped, but the unzipped version is retained.

# Function to print error messages
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Check for correct number of arguments
if [ "$#" -ne 3 ]; then
    error_exit "Usage: $0 <source_directory> <destination_directory> <compression_name>"
fi

# Assign arguments to variables
SOURCE_DIR="$1"
DEST_DIR="$2"
COMPRESSION_NAME="$3"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    error_exit "Source directory does not exist: $SOURCE_DIR"
fi

# Create destination directory if it does not exist
mkdir -p "$DEST_DIR" || error_exit "Failed to create destination directory: $DEST_DIR"

# Function to determine the dataset and scene name
get_dataset_and_scene() {
    local folder_name="$1"
    local dataset=""
    local scene=""

    # Extract the scene name (between last and second to last underscore)
    scene=$(echo "$folder_name" | awk -F'_' '{print $(NF-1)}')

    # Determine the dataset by searching for the specific keywords in the folder name
    if [[ "$folder_name" =~ 360 ]]; then
        dataset="MipNeRF360"
    elif [[ "$folder_name" =~ db ]]; then
        dataset="DeepBlending"
    elif [[ "$folder_name" =~ blender ]]; then
        dataset="SyntheticNeRF"
    elif [[ "$folder_name" =~ tand ]]; then
        dataset="TanksAndTemples"
    else
        error_exit "Unknown dataset prefix for folder: $folder_name"
    fi

    echo "$dataset/$scene"
}


# Function to copy specific files and directories
copy_files() {
    local src_dir="$1"
    local dest_dir="$2"
    local compression_name="$3"

    # Copy cameras.json
    if [ -f "$src_dir/cameras.json" ]; then
        cp "$src_dir/cameras.json" "$dest_dir" || error_exit "Failed to copy cameras.json"
    fi

    # Copy training_config.yaml
    if [ -f "$src_dir/training_config.yaml" ]; then
        cp "$src_dir/training_config.yaml" "$dest_dir" || error_exit "Failed to copy training_config.yaml"
    fi

    # Copy cfg_args
    if [ -f "$src_dir/cfg_args" ]; then
        cp "$src_dir/cfg_args" "$dest_dir" || error_exit "Failed to copy cfg_args"
    fi

    # Copy files from compression/iteration_30000/<compression_name>/
    local comp_src_dir="$src_dir/compression/iteration_30000/$compression_name"
    if [ -d "$comp_src_dir" ]; then
        local comp_dest_dir="$dest_dir/compression/iteration_30000/$compression_name"
        mkdir -p "$comp_dest_dir" || error_exit "Failed to create directory: $comp_dest_dir"
        find "$comp_src_dir" -maxdepth 1 -type f -exec cp {} "$comp_dest_dir" \; || error_exit "Failed to copy files from $comp_src_dir"
    fi
}

# Iterate over each subdirectory in the source directory
for subdir in "$SOURCE_DIR"/*/; do
    subdir_name=$(basename "$subdir")
    
    # Determine dataset and scene
    dataset_scene=$(get_dataset_and_scene "$subdir_name")
    
    dest_subdir="$DEST_DIR/$dataset_scene"

    # Create destination subdirectory
    mkdir -p "$dest_subdir" || error_exit "Failed to create directory: $dest_subdir"

    # Copy the specified files and directories
    copy_files "$subdir" "$dest_subdir" "$COMPRESSION_NAME"
done

# Zip the entire destination directory without compression
zip_file="$DEST_DIR.zip"
zip -r -0 "$zip_file" "$DEST_DIR" || error_exit "Failed to zip the directory: $DEST_DIR"

echo "Files copied and zipped successfully. Zip file created at: $zip_file"

# Function to create a zip file for each dataset without compression
zip_datasets() {
    local dest_dir="$1"

    # Iterate over each dataset directory within the destination directory
    for dataset_dir in "$dest_dir"/*/; do
        dataset_name=$(basename "$dataset_dir")
        zip_file="$dest_dir/$dataset_name.zip"

        # Zip the contents of each dataset directory without compression
        zip -r -0 "$zip_file" "$dataset_dir" || error_exit "Failed to zip the dataset directory: $dataset_dir"

        echo "Dataset zipped successfully: $zip_file"
    done
}

# Call the function to create individual dataset zips
zip_datasets "$DEST_DIR"
