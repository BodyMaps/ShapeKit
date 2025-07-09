# Organ Segmentation Post-Processing Pipeline

This repository contains a comprehensive post-processing pipeline for medical image segmentation, specifically designed to correct and refine multi-organ segmentations with automated case management tools.

## Overview

The pipeline consists of three main tools that work together to provide a complete post-processing workflow:

1. **`generate_missing_cases.py`** - Case list management and missing case detection
2. **`organ_postprocessing.py`** - Core post-processing algorithm for organ segmentations
3. **`merge_labels.py`** - Merge individual organ files into combined multi-label images

## Quick Start Workflow

### Step 1: Generate Case Lists (Optional)
```bash
# Generate list of missing cases for processing
python generate_missing_cases.py --input /path/to/cases --output missing_cases.txt

# Or generate list of existing cases
python generate_missing_cases.py --input /path/to/cases --output existing_cases.txt --mode existing

# Find missing cases in custom range
python generate_missing_cases.py --input /path/to/cases --output missing_100_500.txt --start 100 --end 500
```

**Key Parameters:**
- `--input`, `-i`: Input directory containing BDMAP case folders (required)
- `--output`, `-o`: Output txt file path for the cases list (required)
- `--mode`, `-m`: Generate 'missing' or 'existing' cases list (default: missing)
- `--start`, `-s`: Starting case number for range (default: 1)
- `--end`, `-e`: Ending case number for range (default: 1000)
- `--debug`, `-d`: Enable debug output

### Step 2: Run Post-Processing
```bash
# Process specific cases from list
python organ_postprocessing.py --input /path/to/cases --case_list missing_cases.txt --output /path/to/output

# Or process all BDMAP cases found in input directory (default behavior)
python organ_postprocessing.py --input /path/to/cases --output /path/to/output

# Control parallel processing with custom number of CPU cores
python organ_postprocessing.py --input /path/to/cases --output /path/to/output --processes 8
```

**Key Parameters:**
- `--input`, `-i`: Input directory containing case folders (required)
- `--output`, `-o`: Output directory for processed results (optional)
- `--processes`, `-p`: Number of parallel processes (default: all CPU cores automatically detected)
- `--case_list`, `-c`: Path to txt file with specific cases to process (optional)


**Default Processing:** When `--case_list` is NOT specified, processes ALL BDMAP cases found in the input directory.

**Parallel Processing:** When `--processes` is NOT specified, automatically uses ALL available CPU cores for maximum performance.

## Complete Workflow Examples

### Example 1: Process Missing Cases Only
```bash
# Step 1: Find missing cases
python generate_missing_cases.py -i /data/raw_cases -o missing.txt

# Step 2: Process only missing cases with controlled parallelization
python organ_postprocessing.py -i /data/raw_cases -c missing.txt -o /data/processed --processes 8
```

### Example 2: Full Dataset Processing
```bash
# Process all cases found in input directory with maximum performance
python organ_postprocessing.py -i /data/cases -o /data/output
```

### Example 3: Targeted Processing with Custom Range
```bash
# Step 1: Generate specific range
python generate_missing_cases.py -i /data/cases -o range_500_600.txt --start 500 --end 600

# Step 2: Process with specific class map and conservative parallelization
python organ_postprocessing.py -i /data/cases -c range_500_600.txt -o /data/output  --processes 
```

## Input/Output Structure

### Expected Input Structure
```
input/BDMAP_00000001/
├── segmentations/
│   ├── liver.nii.gz
│   ├── lung_left.nii.gz
│   ├── lung_right.nii.gz
│   ├── kidney_left.nii.gz
│   └── ...other organs.nii.gz
```

### Generated Output Structure (with --output parameter)
```
output/BDMAP_00000001/
├── combined_labels.nii.gz
├── postprocessing.log
└── segmentations/
    ├── liver.nii.gz (processed)
    ├── lung_left.nii.gz (processed)
    ├── lung_right.nii.gz (processed)
    └── ...other organs.nii.gz
```

### Generated Output Structure (without --output parameter)
```
input/BDMAP_00000001/
├── segmentations/ (original)
│   ├── liver.nii.gz
│   └── ...
└── after_processing/
    ├── combined_labels.nii.gz
    ├── postprocessing.log
    └── segmentations/
        ├── liver.nii.gz (processed)
        └── ...other organs.nii.gz
```

## Core Features

### Post-Processing Corrections
- **Liver segmentation**: Anatomical filtering with fragmentation control 
- **Lung overlap resolution**: Fixes conflicts between left and right lung segmentations
- **Femur left/right classification**: Ensures proper labeling based on anatomical position
- **Prostate region filtering**: Identifies correct prostate region
- **Pancreas oversegmentation correction**: Merges excessive connected components
- **Small component noise removal**: Removes artifacts and small disconnected regions


### Processing Logs
- Individual case logs: `postprocessing.log` in each case directory
- Summary logs: `postprocessing_summary.log` in output root directory

## Additional Tool: merge_labels.py

### Purpose
Merges individual organ segmentation files into single multi-label images for easier visualization and analysis.

### Usage
```bash

# Create combined multi-label images
python merge_labels.py --input_dir /path/to/output --class_map pants

# Use specific class map
python merge_labels.py --input_dir /path/to/output --class_map 1.1
```

### Key Parameters
- `--input_dir`, `-i`: Directory containing processed case folders (required)
- `--class_map`, `-c`: Class mapping to use (choices: '1.1', 'pants', required)


### Output
For each case, generates:
- `combined_labels.nii.gz`: Multi-label 3D volume
- `label_mapping.json`: JSON file mapping label IDs to organ names

## Technical Details

### Processing Algorithm Overview
1. **Dynamic organ discovery**: Scans segmentation files to build organ maps
2. **Liver fragmentation control**: Uses spatial positioning to handle incorrectly identified liver positions
3. **Anatomical filtering**: Uses spatial relationships between organs for validation
4. **Connected component analysis**: Identifies and processes individual regions
5. **Noise removal**: Removes small components and merges larger noise regions
6. **Special organ processing**: Custom algorithms for lung, femur, pancreas, prostate

### Adaptive Processing
- Automatically adapts to available organs in dataset
- Missing organs are skipped without errors
- Preserves unprocessed organs in output
- Flexible class map system for different datasets
- Intelligent fragmentation handling for severely fragmented organs

### Logging and Monitoring
- Individual case logs: `postprocessing.log` in each case directory
- Summary logs: `postprocessing_summary.log` in output root
- Debug mode available for detailed troubleshooting
- Progress tracking for large batch processing
- Fragmentation statistics and volume preservation reporting


