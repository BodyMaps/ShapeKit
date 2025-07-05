# Organ Segmentation Post-Processing

This repository contains a post-processing pipeline for medical image segmentation, specifically designed to correct and refine multi-organ segmentations.

## Overview

The post-processing tools address several common issues in automated organ segmentation:

1. **Liver segmentation correction** - Removes liver segmentations that are out of anatomical range
2. **Lung overlap correction** - Resolves conflicts between left and right lung segmentations
3. **Femur left/right classification** - Ensures proper labeling of left and right femur
4. **Prostate filtering** - Identifies the correct prostate region based on anatomical position
5. **Pancreas refinement** - Handles pancreas oversegmentation by merging connected components
6. **Small component removal** - Removes noise and keeps only significant organ regions



## Usage
1. Run the script from the command line with input and output directories:
   ```bash
   python -W ignore organ_postprocessing.py --input /mnt/bodymaps/mask_only/JuMa/JuMa/ --output /mnt/T9/temp_data_to_delete_very_soon/zzhou82/JuMa_nwang52 --processes 12
   ```

2. Alternatively, you can run without specifying an output directory, which will save results in an `after_processing` subdirectory within each case directory:
   ```bash
   python organ_postprocessing.py --input /path/to/input_cases
   ```

3. To use multiple CPU cores for parallel processing of cases:
   ```bash
   python organ_postprocessing.py --input E:\zhuomian\after_processing\JuMaMiniSC\JuMaMiniSC --output E:\zhuomian\after_processing\JuMaMiniSC\results --processes 4
   ```

4. You can also modify the script directly:
   ```python
   if __name__ == "__main__":
       args = parse_arguments()
       input_dir = "/path/to/input_cases"  # Override with your input path
       output_dir = "/path/to/output_dir"  # Override with your output path
       num_processes = 8                   # Number of CPU cores to use
       process_all_cases(input_dir, output_dir, num_processes)
       logging.info("Post-processing completed")
   ```

## File Structure

Expected input directory structure:

```
intput/case_folder/
├── segmentations/
│   ├── liver.nii.gz
│   ├── lung_left.nii.gz
│   ├── lung_right.nii.gz
│   ├── kidney_left.nii.gz
│   └── ...other organs
```

- Each case should be in its own subdirectory.
- The subdirectory name does not affect processing and can be anything.

## Functions

### `build_organ_maps(seg_dir)`

Dynamically builds organ label mapping and processing parameters.

**Parameters**:

- `seg_dir`: Directory path of the segmentation files.

**Returns**:

- `organ_label_map`: A dictionary mapping organ names to label IDs
- `organ_processing_params`: A dictionary of organ processing parameters

### `remove_small_components(organ_mask, combined_labels, current_label, min_size_ratio, min_merge_ratio, ORGAN_LABEL_MAP)`

Removes small connected regions in organ segmentation and merges larger noise regions into adjacent organs.

**Parameters**:

- `organ_mask`: Binary mask of the organ
- `combined_labels`: Complete label image
- `current_label`: Current organ label being processed
- `min_size_ratio`: Minimum volume ratio relative to the largest connected component
- `min_merge_ratio`: Minimum volume ratio to consider merging
- `ORGAN_LABEL_MAP`: Organ label mapping

### `fix_lung_overlap(lung_left_seg, lung_right_seg, seg_dir)`

Fixes the overlap issue between left and right lung segmentations, determining the midline based on the aorta's position.

**Parameters**:

- `lung_left_seg`: Left lung segmentation mask
- `lung_right_seg`: Right lung segmentation mask
- `seg_dir`: Directory path of the segmentation files

**Returns**:

- Fixed left and right lung mask pair

### `process_femur(combined_labels, bladder_seg, ORGAN_LABEL_MAP)`

Handles the mixing of left and right femurs and filters out noise based on bladder position.

**Parameters**:

- `combined_labels`: Merged label image of all organs
- `bladder_seg`: Bladder segmentation mask
- `ORGAN_LABEL_MAP`: Organ label mapping

**Returns**:

- Corrected label image

### `process_pancreas(combined_labels, ORGAN_LABEL_MAP)`

Special processing for pancreas segmentation, merging based on the positional relationship of connected regions to the lungs and stomach.

**Parameters**:

- `combined_labels`: Merged label image of all organs
- `ORGAN_LABEL_MAP`: Organ label mapping

**Returns**:

- Corrected label image

### `fix_liver_segmentation(case_path: str) -> None`

Performs comprehensive post-processing on a single case, including liver segmentation correction. This function is the core of the entire processing workflow, executing the following steps:

1. Dynamically builds organ label mapping and processing parameters
2. Loads all organ segmentations and creates a combined label image
3. Fixes the overlap issue between left and right lungs
4. Filters liver segmentation based on anatomical location:
   - Applies horizontal constraint to the liver using the x-axis range of the lungs (keeps only the liver regions overlapping with the lung's x-axis range)
   - Applies vertical constraint to the liver using the z-axis positions of the right lung and bladder (ensures the liver is between the right lung and bladder)
5. Removes small connected regions for each organ
6. Processes the classification and filtering of left and right femurs
7. Special processing for the pancreas to address over-segmentation issues
8. Saves the processed results and generates logs

**Parameters**:

- `case_path`: The path to the case directory.

**Returns**:

- `bool`: Returns True if processing is successful, False otherwise.

### `process_all_cases(cases_directory: str) -> None`

Processes all cases in the given directory and generates a comprehensive processing report.

**Parameters**:

- `cases_directory`: The path containing all case subdirectories


## Output

The processed segmentation results will be saved in the `after_processing` subdirectory of each case directory, for example:

```
output/case_folder/
├── segmentations/
│   ├── liver.nii.gz
│   ├── lung_left.nii.gz
│   ├── lung_right.nii.gz
│   ├── kidney_left.nii.gz
│   └── ...other organs
├── postprocessing.log
```

Processing logs are recorded in:
- `postprocessing.log` file in each case directory (records the processing steps for individual cases)
- `postprocessing_summary.log` file in the root directory (records overall processing progress and statistics)

## Processing Parameters

The algorithm uses the following parameters to control processing behavior:

- **Small Component Removal**:
  - `min_size_ratio`: Default value 0.25, the minimum volume ratio relative to the largest connected component
  - `min_merge_ratio`: Default value 0.075, the minimum volume ratio to consider merging into other organs

- **Liver Position Constraints**:
  - X-axis and Z-axis constraints based on lung and bladder positions
  - Lung volume threshold: 10000
  - Bladder volume threshold: 10000

## Additional Tools

### Merge Labels Tool

The repository also includes a tool (`merge_labels.py`) for merging individual organ segmentation files into a single multi-label image file based on predefined classification maps from `class_maps.py`.

#### Usage

Simple usage with predefined class maps:

```bash
python merge_labels.py --input_dir /path/to/cases --class_map 1.1
```

#### Command Line Arguments

- `--input_dir`, `-i`: Input directory containing case folders with segmentations
- `--class_map`, `-c`: Class mapping to use (choices: "1.1" or "pants")
- `--list_maps`, `-l`: List available class maps and exit

#### Predefined Classification Maps

The class maps are defined in `class_maps.py` and include:

1. **1.1**: 25 organs including major abdominal and pelvic organs (AbdomenAtlas 1.1)
2. **pants**: 28 organs including detailed pancreatic structures (AbdomenAtlas PANTS)

#### Example

```bash
# List available class maps
python merge_labels.py --list_maps

# Use default AbdomenAtlas 1.1 classification
python merge_labels.py --input_dir ./processed_cases --class_map 1.1

# Use PANTS classification
python merge_labels.py --input_dir ./processed_cases --class_map pants
```

#### Output

The tool generates the following output for each case directly in the case directory:

- `combined_labels.nii.gz`: A single 3D volume where each voxel value corresponds to an organ ID
- `label_mapping.json`: A JSON file documenting the mapping between label IDs and organ names

This allows for compact storage and simpler visualization of the complete segmentation.


