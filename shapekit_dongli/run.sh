#!/bin/bash

# Get current timestamp in YYYY-MM-DD_HH-MM-SS format
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Define log filename with timestamp
LOG_FILE="result_${TIMESTAMP}.log"

# Save all output to result.log
exec > "$LOG_FILE" 2>&1

# echo "Running post_processing.py for PanTSV2..."
# python post_processing.py \
#   --source_dir data/PanTSV2 \
#   --target_dir data/PanTSV2_processed \
#   --class_map class_map_abdomenatlas_pants \
#   --n_jobs 4

# echo "Running post_processing.py for PanTS..."
# python post_processing.py \
#   --source_dir data/PanTS \
#   --target_dir data/PanTS_processed \
#   --class_map class_map_abdomenatlas_pants \
#   --n_jobs 4

# echo "Running post_processing.py for JuMaMini_noCT..."
# python post_processing.py \
#   --source_dir data/JuMaMini_noCT \
#   --target_dir data/JuMaMini_noCT_processed \
#   --class_map class_map_abdomenatlas_1_1 \
#   --n_jobs 3

echo "Running post_processing.py for JuMaMiniSC..."
python post_processing.py \
  --source_dir data/JuMaMiniSC \
  --target_dir data/JuMaMiniSC_processed \
  --class_map class_map_abdomenatlas_1_1 \
  --n_jobs 4

echo "Processing complete. Log saved to $LOG_FILE"