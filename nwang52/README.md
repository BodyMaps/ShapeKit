# Organ Segmentation Post-Processing Pipeline

This repository contains a comprehensive post-processing pipeline for medical image segmentation, specifically designed to correct and refine multi-organ segmentations with automated case management tools.

## Overview

The pipeline consists of four main tools that work together to provide a complete post-processing workflow:

1. **`generate_missing_cases.py`** - Case list management and missing case detection
2. **`organ_postprocessing.py`** - Core post-processing algorithm for organ segmentations
3. **`merge_labels.py`** - Merge individual organ files into combined multi-label images
4. **`cleanup_incomplete_cases.py`** - Clean up incomplete or failed processing results

## Quick Start Workflow

### Step 1: Generate Case Lists (Optional)
```bash
# Generate list of missing cases for processing
python generate_missing_cases.py --input /path/to/cases --output missing_cases.txt

# Or generate list of existing cases
python generate_missing_cases.py --input /path/to/cases --output existing_cases.txt --mode existing
```

### Step 2: Run Post-Processing
```bash
# Process specific cases from list
python organ_postprocessing.py --input /path/to/cases --case_list missing_cases.txt --output /path/to/output

# Or process all cases in default range (BDMAP_00000001 to BDMAP_00001000)
python organ_postprocessing.py --input /path/to/cases --output /path/to/output
```

### Step 3: Merge Individual Segmentations (Optional)
```bash
# Create combined multi-label images
python merge_labels.py --input_dir /path/to/output --class_map all
```

### Step 4: Clean Up Incomplete Results (Optional)
```bash
# Remove folders without combined_labels.nii.gz
python cleanup_incomplete_cases.py --input /path/to/output --dry-run
```

## Detailed Tool Documentation

## 1. generate_missing_cases.py

### Purpose
Analyzes your dataset to identify missing cases and generates targeted processing lists.

### Usage
```bash
python generate_missing_cases.py [OPTIONS]
```

### Arguments
- `--input`, `-i`: Input directory containing BDMAP case folders (required)
- `--output`, `-o`: Output txt file path for the cases list (required)
- `--mode`, `-m`: Generate 'missing' or 'existing' cases list (default: missing)
- `--start`, `-s`: Starting case number for range (default: 1)
- `--end`, `-e`: Ending case number for range (default: 1000)
- `--debug`, `-d`: Enable debug output

### Examples
```bash
# Find missing cases in default range (1-1000)
python generate_missing_cases.py -i /data/cases -o missing.txt

# Find missing cases in custom range
python generate_missing_cases.py -i /data/cases -o missing_100_500.txt --start 100 --end 500

# List all existing cases
python generate_missing_cases.py -i /data/cases -o existing.txt --mode existing

# Enable debug mode
python generate_missing_cases.py -i /data/cases -o missing.txt --debug
```

### Output Format
The generated txt file contains one case name per line:
```
BDMAP_00000001
BDMAP_00000005
BDMAP_00000012
...
```

## 2. organ_postprocessing.py

### Purpose
Core post-processing tool that corrects common segmentation issues including:
- Liver segmentation anatomical filtering
- Lung overlap resolution
- Femur left/right classification
- Prostate region filtering
- Pancreas oversegmentation correction
- Small component noise removal

### Usage
```bash
python organ_postprocessing.py [OPTIONS]
```

### Arguments
- `--input`, `-i`: Input directory containing case folders (required)
- `--output`, `-o`: Output directory for processed results (optional)
- `--processes`, `-p`: Number of parallel processes (default: all CPU cores)
- `--case_list`, `-c`: Path to txt file with specific cases to process (optional)
- `--class_map`, `-m`: Class map for processing scope (choices: '1.1', 'pants', 'all', default: 'all')
- `--debug`, `-d`: Enable debug output

### Class Maps
The tool supports three class maps that determine which organs are processed:

#### "all" (Default) - 34 organs
Complete mapping including all organs from AbdomenAtlas 1.1 and PANTS datasets:
- All major abdominal and pelvic organs
- Detailed pancreatic structures (body, head, tail, duct, tumor)
- Both vascular naming conventions (celiac_trunk and celiac_artery)

#### "1.1" - 25 organs
Standard AbdomenAtlas 1.1 mapping:
- Major organs: liver, pancreas, spleen, stomach, kidneys
- Vascular: aorta, postcava, portal_vein_and_splenic_vein
- Pelvic: bladder, prostate, rectum
- Skeletal: femur_left, femur_right

#### "pants" - 28 organs
PANTS dataset mapping with detailed pancreatic structures:
- Standard abdominal organs
- Detailed pancreas: pancreas_body, pancreas_head, pancreas_tail
- Pathology support: pancreatic_tumor
- Ductal structures: pancreatic_duct, common_bile_duct

### Processing Parameters
The algorithm uses these key parameters:

#### Small Component Removal
- `min_size_ratio`: 0.25 (25% of largest component)
- `min_merge_ratio`: 0.075 (7.5% threshold for merging)

#### Anatomical Constraints
- Lung volume threshold: 10,000 voxels
- Bladder volume threshold: 10,000 voxels
- Liver x-axis tolerance: 10% of lung range
- Femur z-axis distance multiplier: 10x minimum distance

### Examples
```bash
# Basic processing with case list
python organ_postprocessing.py -i /data/cases -c missing.txt -o /data/output

# Use specific class map
python organ_postprocessing.py -i /data/cases --class_map 1.1 -o /data/output

# Parallel processing with 8 cores
python organ_postprocessing.py -i /data/cases -p 8 -o /data/output

# Process all cases in default range
python organ_postprocessing.py -i /data/cases -o /data/output

# Debug mode
python organ_postprocessing.py -i /data/cases -c test.txt --debug
```

### Input Structure
Expected directory structure:
```
input/BDMAP_00000001/
├── segmentations/
│   ├── liver.nii.gz
│   ├── lung_left.nii.gz
│   ├── lung_right.nii.gz
│   ├── kidney_left.nii.gz
│   └── ...other organs.nii.gz
```

### Output Structure
Generated output structure:
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

**Note**: Each case gets its own folder in the output directory. The folder name matches the input case folder name (e.g., BDMAP_00000001).

## 3. merge_labels.py

### Purpose
Merges individual organ segmentation files into single multi-label images for easier visualization and analysis.

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

## 4. cleanup_incomplete_cases.py

### Purpose
Identifies and removes case folders that failed processing or are missing required output files.

### Usage
```bash
python cleanup_incomplete_cases.py [OPTIONS]
```

### Arguments
- `--input`, `-i`: Directory containing case folders to check (required)
- `--dry-run`, `-d`: Show what would be deleted without actually deleting
- `--list-only`, `-l`: Only list incomplete cases without deletion
- `--verbose`, `-v`: Enable verbose logging

### Examples
```bash
# Check what would be deleted (safe)
python cleanup_incomplete_cases.py -i /data/output --dry-run

# List incomplete cases only
python cleanup_incomplete_cases.py -i /data/output --list-only

# Actually delete incomplete cases (WARNING: permanent deletion)
python cleanup_incomplete_cases.py -i /data/output

# Verbose mode
python cleanup_incomplete_cases.py -i /data/output --list-only --verbose
```

### Features
- **Safe Mode**: Use `--dry-run` to preview changes without deletion
- **List Mode**: Use `--list-only` to only view incomplete cases
- **Detection Criteria**: Identifies cases missing `combined_labels.nii.gz` files
- **Batch Operations**: Processes all BDMAP cases in the directory

## Complete Workflow Examples

### Example 1: Process Missing Cases Only
```bash
# Step 1: Find missing cases
python generate_missing_cases.py -i /data/raw_cases -o missing.txt

# Step 2: Process only missing cases
python organ_postprocessing.py -i /data/raw_cases -c missing.txt -o /data/processed

# Step 3: Create combined labels
python merge_labels.py -i /data/processed --class_map all

# Step 4: Clean up any failures
python cleanup_incomplete_cases.py -i /data/processed --dry-run
```

### Example 2: Full Dataset Processing
```bash
# Step 1: Process all cases in range
python organ_postprocessing.py -i /data/cases -o /data/output --processes 16

# Step 2: Merge labels
python merge_labels.py -i /data/output --class_map all

# Step 3: Clean up failures
python cleanup_incomplete_cases.py -i /data/output --list-only
```

### Example 3: Targeted Processing with Custom Range
```bash
# Step 1: Generate specific range
python generate_missing_cases.py -i /data/cases -o range_500_600.txt --start 500 --end 600

# Step 2: Process with specific class map
python organ_postprocessing.py -i /data/cases -c range_500_600.txt -o /data/output --class_map 1.1

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

### Performance Optimization

- Use `--processes` parameter to control CPU usage
- Process in batches using custom case lists
- Use appropriate class maps to avoid processing unnecessary organs
- Monitor disk space as output can be substantial

### File Permissions

Ensure write permissions for:
- Output directories
- Log file locations
- Temporary processing space

## Technical Details

### Processing Algorithm Overview
1. **Dynamic organ discovery**: Scans segmentation files to build organ maps
2. **Anatomical filtering**: Uses spatial relationships between organs for validation
3. **Connected component analysis**: Identifies and processes individual regions
4. **Noise removal**: Removes small components and merges larger noise regions
5. **Special organ processing**: Custom algorithms for lung, femur, pancreas, prostate

### Adaptive Processing
- Automatically adapts to available organs in dataset
- Missing organs are skipped without errors
- Preserves unprocessed organs in output
- Flexible class map system for different datasets

### Logging and Monitoring
- Individual case logs: `postprocessing.log` in each case directory
- Summary logs: `postprocessing_summary.log` in output root
- Debug mode available for detailed troubleshooting
- Progress tracking for large batch processing

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


