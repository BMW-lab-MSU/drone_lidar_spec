#!/bin/bash

# Variables
IMAGES_PATH="/home/d86p233/Desktop/BMW-spec/merged_data/noboost_prop-only/images"
OUTPUT_PATH="noboost_prop-only.json"
H5_JSON="SFh5s.json"
IMAGE_JSON="$OUTPUT_PATH"
COMPARISON_OUTPUT="noboost_prop-only-comparisons.json"
METRICS_JSON="$COMPARISON_OUTPUT"
RESULTS_JSON="noboost_prop-only-results.json"

# Run the sliding window script
python slide_windows.py --images_path "$IMAGES_PATH" --output_path "$OUTPUT_PATH"

# Run the frequency comparison script
python compare_frequencies.py --h5_json "$H5_JSON" --image_json "$IMAGE_JSON" --output_path "$COMPARISON_OUTPUT"

# Run the evaluation metrics script
python calc_eval_metrics.py --json_path "$METRICS_JSON" --output_path "$RESULTS_JSON"
