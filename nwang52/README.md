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
- `--class_map`, `-m`: Class map for processing scope (choices: '1.1', 'pants', 'all', default: 'all')

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
python organ_postprocessing.py -i /data/cases -c range_500_600.txt -o /data/output --class_map 1.1 --processes 6
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
- **Liver segmentation**: Anatomical filtering with fragmentation control (keeps only 20 largest components when >50 components detected)
- **Lung overlap resolution**: Fixes conflicts between left and right lung segmentations
- **Femur left/right classification**: Ensures proper labeling based on anatomical position
- **Prostate region filtering**: Identifies correct prostate region
- **Pancreas oversegmentation correction**: Merges excessive connected components
- **Small component noise removal**: Removes artifacts and small disconnected regions

### Class Maps
- **"all" (Default)**: 34 organs - comprehensive mapping from AbdomenAtlas 1.1 and PANTS
- **"1.1"**: 25 organs - standard AbdomenAtlas 1.1 mapping
- **"pants"**: 28 organs - PANTS dataset with detailed pancreatic structures

### Processing Logs
- Individual case logs: `postprocessing.log` in each case directory
- Summary logs: `postprocessing_summary.log` in output root directory

## Additional Tool: merge_labels.py

### Purpose
Merges individual organ segmentation files into single multi-label images for easier visualization and analysis.

### Usage
```bash
# List available class maps
python merge_labels.py --list_maps

# Create combined multi-label images
python merge_labels.py --input_dir /path/to/output --class_map all

# Use specific class map
python merge_labels.py --input_dir /path/to/output --class_map 1.1
```

### Key Parameters
- `--input_dir`, `-i`: Directory containing processed case folders (required)
- `--class_map`, `-c`: Class mapping to use (choices: '1.1', 'pants', 'all', required)
- `--list_maps`, `-l`: List available class maps and exit

### Output
For each case, generates:
- `combined_labels.nii.gz`: Multi-label 3D volume
- `label_mapping.json`: JSON file mapping label IDs to organ names

### Usage
```bash
python merge_labels.py [OPTIONS]
```

### Arguments
- `--input_dir`, `-i`: Directory containing processed case folders (required)
- `--class_map`, `-c`: Class mapping to use (choices: '1.1', 'pants', 'all', required)
- `--list_maps`, `-l`: List available class maps and exit

### Examples
```bash
# List available class maps
python merge_labels.py --list_maps

# Merge using comprehensive class map
python merge_labels.py -i /data/output --class_map all

# Use specific class map
python merge_labels.py -i /data/output --class_map 1.1
```

### Output
For each case, generates:
- `combined_labels.nii.gz`: Multi-label 3D volume
- `label_mapping.json`: JSON file mapping label IDs to organ names

## Complete Workflow Examples

### Example 1: Process Missing Cases Only
```bash
# Step 1: Find missing cases
python generate_missing_cases.py -i /data/raw_cases -o missing.txt

# Step 2: Process only missing cases with controlled parallelization
python organ_postprocessing.py -i /data/raw_cases -c missing.txt -o /data/processed --processes 8

# Step 3: Create combined labels
python merge_labels.py -i /data/processed --class_map all
```

### Example 2: Full Dataset Processing
```bash
# Step 1: Process all cases found in input directory with maximum performance
python organ_postprocessing.py -i /data/cases -o /data/output --processes 16

# Step 2: Merge labels
python merge_labels.py -i /data/output --class_map all
```

### Example 3: Targeted Processing with Custom Range
```bash
# Step 1: Generate specific range
python generate_missing_cases.py -i /data/cases -o range_500_600.txt --start 500 --end 600

# Step 2: Process with specific class map and conservative parallelization
python organ_postprocessing.py -i /data/cases -c range_500_600.txt -o /data/output --class_map 1.1 --processes 6

# Step 3: Merge with matching class map
python merge_labels.py -i /data/output --class_map 1.1
```

## Output Management

### When using --output parameter:
```
output_dir/BDMAP_00000001/
├── combined_labels.nii.gz
├── postprocessing.log
└── segmentations/
    ├── liver.nii.gz (processed)
    ├── lung_left.nii.gz (processed)
    ├── lung_right.nii.gz (processed)
    ├── kidney_left.nii.gz (processed)
    └── ...other organs.nii.gz
```

### When NOT using --output parameter (in-place processing):
```
input_dir/BDMAP_00000001/
├── segmentations/ (original)
│   ├── liver.nii.gz
│   ├── lung_left.nii.gz
│   └── ...
└── after_processing/
    ├── combined_labels.nii.gz
    ├── postprocessing.log
    └── segmentations/
        ├── liver.nii.gz (processed)
        ├── lung_left.nii.gz (processed)
        └── ...other organs.nii.gz
```

Processing logs are recorded in:
- `postprocessing.log` file in each case's output directory (records the processing steps for individual cases)
- `postprocessing_summary.log` file in the root output directory (records overall processing progress and statistics)

## Troubleshooting

### Common Issues

1. **No cases found**: Check that case folders start with 'BDMAP_' and contain 'segmentations' subdirectory
2. **Processing failures**: Check individual case logs in `postprocessing.log` files
3. **Memory issues**: Reduce number of parallel processes with `-p` parameter
4. **Missing organs**: Use `--debug` flag to see which organs are skipped
5. **Liver fragmentation warnings**: Normal for cases with >50 connected components; only largest 20 are kept

**Default Processing:**
- When no `--case_list` is specified, ALL BDMAP cases in the input directory are processed
- Cases are automatically discovered and sorted by case number
- Use `--debug` flag to see which cases are found and will be processed

### Performance Optimization

- **Parallel processing**: Use `--processes` parameter to control CPU usage
- **Batch processing**: Process in batches using custom case lists
- **Class map selection**: Use appropriate class maps to avoid processing unnecessary organs
- **Storage considerations**: Monitor disk space as output can be substantial
- **Memory management**: Reduce parallel processes if experiencing memory pressure

### File Permissions

Ensure write permissions for:
- Output directories
- Log file locations
- Temporary processing space

## Technical Details

### Processing Algorithm Overview
1. **Dynamic organ discovery**: Scans segmentation files to build organ maps
2. **Liver fragmentation control**: Keeps only 20 largest components when >50 components detected
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

## Class Maps

The tool supports three predefined class maps for different organ sets:

### 1. "all" (Default) - Comprehensive Map (34 organs)
A complete mapping that includes all organs from both AbdomenAtlas 1.1 and PANTS datasets:
- All major abdominal and pelvic organs
- Detailed pancreatic structures (body, head, tail, duct, tumor)
- Both vascular naming conventions (celiac_trunk and celiac_artery)
- Complete organ coverage for maximum compatibility

### 2. "1.1" - AbdomenAtlas 1.1 (25 organs)
Standard AbdomenAtlas mapping including:
- Major abdominal organs (liver, pancreas, spleen, stomach, kidneys)
- Vascular structures (aorta, postcava, portal_vein_and_splenic_vein)
- Pelvic organs (bladder, prostate, rectum)
- Skeletal structures (femur_left, femur_right)

### 3. "pants" - PANTS Dataset (28 organs)
Enhanced mapping with detailed pancreatic structures:
- All standard abdominal organs
- Detailed pancreatic segmentation (pancreas_body, pancreas_head, pancreas_tail)
- Pancreatic pathology support (pancreatic_tumor)
- Ductal structures (pancreatic_duct, common_bile_duct)

### Adaptive Processing

The tool automatically adapts to available organs in your dataset:
- **Missing organs**: If an organ from the class map is not found in segmentations, it will be skipped without errors
- **Extra organs**: Organs present in segmentations but not in the class map will be preserved but not processed
- **Flexible compatibility**: The "all" class map provides maximum compatibility across different datasets


