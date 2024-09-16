#!/bin/bash

# This script downloads and evaluates the Self-Organizing Gaussians (SOGS) scenes.
# It will download and unpack the scenes from the SOGS repository,
# decompress the models to .ply, render the evaluation images, compute the metrics, and collect the evaluation results.

# It supports two datasets: one with spherical harmonics ("w/ SH", Baseline) and one without ("w/o SH").
#
# Usage: ./download_eval.sh <destination_directory> [source_search_path1 source_search_path2 ...]
#        e.g. ./download_eval.sh /data/sogs_results /data/DeepBlending /data/Blender /data/MipNerf360 /data/MipNerf360_extra /data/TandT
# 
# Output will be saved as .csv files in the destination directory, under results/

set -euo pipefail

# Add the Self-Organizing Gaussians code folder of where the script is located to PYTHONPATH
CODE_DIR="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
echo "CODE_DIR: $CODE_DIR"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

# Ensure at least one argument is passed (data directory)
if [ $# -lt 1 ]; then
  echo "Usage: $0 <data_dir> [source_search_path1 source_search_path2 ...]"
  exit 1
fi

DEST_DIR="$1"
shift  # Remove the first argument

# Remaining arguments are scene source search paths (if any)
SEARCH_PATHS=("$@")

echo "Destination directory: $DEST_DIR"
echo "Dataset source search paths: ${SEARCH_PATHS[@]}"

# Function to find a scene in the provided search paths
find_scene_in_paths() {
    local scene_name="$1"

    for search_path in "${SEARCH_PATHS[@]}"; do
        if [ -d "$search_path/$scene_name" ]; then
            echo "$(readlink -f "$search_path/$scene_name")"  # Return the absolute path
            return 0
        fi
    done

    echo "Scene directory '$scene_name' not found in search paths: ${SEARCH_PATHS[*]}" >&2
    return 1
}

# List of required scenes
SCENE_LIST=(drjohnson playroom bicycle bonsai counter flowers garden kitchen room stump treehill chair drums ficus hotdog lego materials mic ship train truck)

# Check that all required scenes can be found
MISSING_SCENES=()

for SCENE in "${SCENE_LIST[@]}"; do
    if ! find_scene_in_paths "$SCENE" >/dev/null 2>&1; then
        MISSING_SCENES+=("$SCENE")
    fi
done

if [ ${#MISSING_SCENES[@]} -ne 0 ]; then
    echo "The following scene directories were not found in the search paths: ${MISSING_SCENES[*]}"
    echo "Please add the dataset source directories as additional arguments to this script."
    exit 1
fi

mkdir -p "$DEST_DIR"

# Function to download and process a dataset
process_dataset() {
    local DATASET_URL="$1"
    local DATA_SUBDIR="$2"
    local SUBMETHOD="$3"
    local COMPRESSION_DIR_NAME="$4"

    # Extract the ZIP file name from the URL
    local ZIP_FILE_NAME="${DATASET_URL##*/}"

    # Download and extract the dataset
    cd "$DEST_DIR"
    echo "Downloading $ZIP_FILE_NAME..."
    curl -C - -# -L "$DATASET_URL" -o "$ZIP_FILE_NAME" || { echo "Download failed"; exit 1; }
    unzip -n -q "$ZIP_FILE_NAME" || { echo "Failed to unzip"; exit 1; }
    echo "Download and extraction complete for $ZIP_FILE_NAME."

    # Process each dataset and scene
    local DATASET_PATH="$DEST_DIR/$DATA_SUBDIR"
    cd "$DATASET_PATH"

    for DATASET in */; do
      DATASET=${DATASET%/}
      echo "Dataset: $DATASET"

      for DATASET_SCENE in "$DATASET"/*/; do
        DATASET_SCENE=${DATASET_SCENE%/}
        SCENE=$(basename "$DATASET_SCENE")
        COMPRESSED_MODEL_PATH="$DATASET_PATH/$DATASET_SCENE/compression/iteration_30000/$COMPRESSION_DIR_NAME/"

        SCENE_SOURCE_PATH=$(find_scene_in_paths "$SCENE") || { echo "Scene '$SCENE' not found, please add its source directory to the search paths."; exit 1; }
        echo "Scene: $DATASET_SCENE, Source path: $SCENE_SOURCE_PATH"

        echo "  Decompressing $DATASET_SCENE to 3DGS .ply"
        python "${CODE_DIR}/compression/decompress.py" \
          --compressed_model_path "${COMPRESSED_MODEL_PATH}"

        echo "  Rendering eval images for $DATASET_SCENE"
        python "${CODE_DIR}/render.py" \
          --source_path "${SCENE_SOURCE_PATH}" \
          --model_path "${COMPRESSED_MODEL_PATH}/decompressed_model" \
          --skip_train \
          --eval \
          --data_device cuda \
          --disable_xyz_log_activation

        echo "  Computing metrics for $DATASET_SCENE"
        python "${CODE_DIR}/metrics.py" \
          --model_path "${COMPRESSED_MODEL_PATH}/decompressed_model"

        echo "  Collecting evaluation results for $DATASET_SCENE"
        python "${CODE_DIR}/eval/collect_eval_per_scene.py" \
          --output-dir "${DEST_DIR}" \
          --dataset "${DATASET}" \
          --scene "${SCENE}" \
          --model-path "${COMPRESSED_MODEL_PATH}" \
          --submethod "${SUBMETHOD}"
      done
    done
}

# ECCV Baseline, with SH
DATASET1_URL="https://github.com/fraunhoferhhi/Self-Organizing-Gaussians/releases/download/eccv-2024-data/Scenes_SOGS_ECCV_with_SH.zip"
DATASET1_SUBDIR="results_SOGS_ECCV/with_SH"
DATASET1_SUBMETHOD="Baseline"
DATASET1_COMPRESSION_DIR="jxl_quant_sh"

# ECCV w/o SH
DATASET2_URL="https://github.com/fraunhoferhhi/Self-Organizing-Gaussians/releases/download/eccv-2024-data/Scenes_SOGS_ECCV_without_SH.zip"
DATASET2_SUBDIR="results_SOGS_ECCV/without_SH"
DATASET2_SUBMETHOD="w/o SH"
DATASET2_COMPRESSION_DIR="jxl_quant"

process_dataset "$DATASET1_URL" "$DATASET1_SUBDIR" "$DATASET1_SUBMETHOD" "$DATASET1_COMPRESSION_DIR"

process_dataset "$DATASET2_URL" "$DATASET2_SUBDIR" "$DATASET2_SUBMETHOD" "$DATASET2_COMPRESSION_DIR"

echo "All scenes processed."
