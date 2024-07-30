#!/bin/bash

# Variables
IMAGES_PATH="../data/noboost_prop-only-90/val/images"
OUTPUT_PATH="noboost_prop-only-90-val.json"
H5_JSON="SFh5s3.json"
IMAGE_JSON="noboost_prop-only-90-val.json"
COMPARISON_OUTPUT="noboost_prop-only-90-val-comparisons.json"
METRICS_JSON="noboost_prop-only-90-val-comparisons.json"
RESULTS_JSON="noboost_prop-only-90-val-results.json"

# Run the sliding window script
python slide_windows.py --images_path "$IMAGES_PATH" --output_path "$OUTPUT_PATH"

# Run the frequency comparison script
python compare_frequencies.py --h5_json "$H5_JSON" --image_json "$IMAGE_JSON" --output_path "$COMPARISON_OUTPUT"

# Run the evaluation metrics script
python calc_eval_metrics.py --json_path "$METRICS_JSON" --output_path "$RESULTS_JSON"
