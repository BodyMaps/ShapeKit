# Atlas-Based Registration Solution for Vertebrae PostProcessing

This guide explains how to use the atlas-based registration solution for vertebrae label refinement integrated into ShapeKit.

## Overview

The atlas-based registration solution provides a sophisticated approach to refine vertebrae segmentations through deformable atlas registration. It uses ANTsPy to register input vertebrae to a reference atlas and applies intelligent label refinement through:

1. **Cleanup**: Remove small fragments and consolidate noisy components
2. **Registration**: ANTsPy-based rigid, affine, and SyN deformable registration
3. **Relabel**: Assign components based on atlas overlap and spatial distance
4. **Finalize**: Consolidate labels to target count via smart merging/splitting

## Quick Start

### 1. Install Dependencies

The warmup mode requires ANTsPy:

```bash
pip install antspyx
```

Install all ShapeKit requirements:

```bash
pip install -r requirements.txt
```

### 2. Enable Warmup Mode in config.yaml

Edit `config.yaml` and locate the `vertebrae` section:

```yaml
vertebrae:
  processing_mode: "warmup"  # Changed from "legacy"
  target_count: 24
  
  # Provide paths to reference atlas CT and labels
  fixed_reference_ct: "/path/to/atlas/ct.nii.gz"
  fixed_reference_labels: "/path/to/atlas/labels.nii.gz"
  
  warmup_config:
    # ... configuration parameters (see defaults below)
```

### 3. Prepare Atlas Reference

You'll need a reference atlas (fixed CT and labels):

- **Fixed CT**: The reference CT image (typically BDMAP_00000031 for full spine)
- **Fixed Labels**: Reference spine labels with values 1-24 for C1-L5 vertebrae

The atlas labels should be organized as:
- Labels 1-7: Cervical vertebrae (C1-C7, from superior to inferior)
- Labels 8-19: Thoracic vertebrae (T1-T12)
- Labels 20-24: Lumbar vertebrae (L1-L5)

### 4. Run Pipeline

```bash
python main.py \
  --input /path/to/input \
  --output /path/to/output \
  --config config.yaml
```

## Configuration Parameters

### Vertebrae Section

```yaml
vertebrae:
  processing_mode: "warmup" | "legacy"  # Pipeline selection
  target_count: 24  # Expected vertebrae count (7, 12, 24, etc.)
  fixed_reference_ct: "/path/to/atlas/ct.nii.gz"
  fixed_reference_labels: "/path/to/atlas/labels.nii.gz"
```

### Cleanup Configuration

Controls fragment removal and noise consolidation:

```yaml
cleanup:
  min_fragment_voxels: 64  # Minimum voxels for a fragment to survive
  min_fragment_ratio: 0.01  # Minimum ratio to main component size
```

### Registration Configuration

Controls ANTsPy registration stages:

```yaml
registration:
  mask_dilate_mm: 6.0  # Foreground mask dilation (mm)
  crop_margin_voxels: 24  # Margin for image cropping
  affine_mask_metric: "meansquares"  # Similarity metric
  enable_rigid_foreground_stage: true  # Run rigid stage first
  label_weight: 1.0  # Weight for label term in multivariate registration
  use_centroid_anchor: true  # Pre-align using centroid shift
  anchor_axis: "all"  # "all" or "z" for z-axis only anchoring
  centroid_shift_clamp_voxels: 40  # Maximum centroid shift (voxels)
  reg_iterations: [30, 10, 0]  # Iterations per level (coarse, medium, fine)
  grad_step: 0.05  # Gradient step for SyN
  flow_sigma: 4.0  # SyN flow smoothing
  total_sigma: 2.0  # SyN total smoothing
```

### Relabel Configuration

Controls component-to-atlas assignment:

```yaml
relabel:
  min_overlap_voxels: 25  # Minimum overlap to assign based on atlas
  centroid_z_weight: 2.5  # Z-axis weight in distance calculation
  overlap_weight: 0.15  # Weight for overlap term in assignment cost
  connectivity: 3  # Connectivity for component detection (1, 2, or 3)
```

### Finalize Configuration

Controls final label consolidation:

```yaml
finalize:
  min_fragment_voxels: 20  # Minimum fragment size to preserve
  split_size_ratio: 0.35  # Size ratio for split detection
  split_distance_ratio: 1.5  # Distance ratio for split detection
  pair_small_ratio: 0.70  # Threshold for small pair merging
  pair_sum_min_ratio: 0.70  # Min combined volume for merging
  pair_sum_max_ratio: 1.40  # Max combined volume for merging
  nearby_spacing_ratio: 0.80  # Z-spacing tolerance for merging
  max_edit_label: 12  # Don't edit labels above this ID
  low_label_aggressiveness: 0.90  # Aggressiveness for low labels
  high_label_distance_penalty: 1.60  # Penalty for high labels
  target_labels: 24  # Final target label count
  connectivity: 3  # Connectivity for component detection
```

## Vertebrae Count Configurations

The pipeline supports flexible vertebrae counts:

### Full Spine (Default)
```yaml
target_count: 24
# C1-C7 (7) + T1-T12 (12) + L1-L5 (5)
```

### Cervical Only
```yaml
target_count: 7
# C1-C7
```

### Thoracic Only
```yaml
target_count: 12
# T1-T12
```

### Lumbar Only
```yaml
target_count: 5
# L1-L5
```

## Output Files

The warmup pipeline generates detailed audit trails for each processing step:

```
output/{patient_id}/vertebrae/
├── split_components.csv           # Connected component analysis
├── pool_assignment.csv            # Fragment pool reassignment
├── volume_merge_pairs.csv         # Pair merging decisions
├── false_positive_repairs.csv     # False positive corrections
├── enforce_target_count.csv       # Target count enforcement
├── parent_label_assignment.csv    # Label-to-atlas mapping
└── component_assignments.csv      # Component-to-label assignments
```

Each CSV contains detailed logging of decisions made during processing, useful for:
- Debugging processing results
- Understanding label transformations
- Validating pipeline behavior on your data

## Performance Considerations

### Memory Usage
- Per-case memory during registration: ~2-3 GB
- Registration is the most memory-intensive step
- Consider hardware: 8+ GB RAM recommended for parallel processing

### Processing Time
- Typical throughput: 1-2 cases/hour per CPU core
- Registration (SyN) is slowest step: ~30-50 minutes per case
- Cleanup and finalization: ~5-10 minutes per case

### Parallel Processing
ShapeKit automatically parallelizes case processing:

```bash
# Use 8 cores for parallel processing
python main.py --input /path/input --output /path/output --cpu-count 8
```

## Troubleshooting

### ANTsPy Installation Issues

If you encounter ANTsPy installation problems:

```bash
# On macOS, you may need:
pip install antspyx --prefer-binary

# On Linux with CUDA:
pip install antspyx-cuda
```

### Atlas Path Errors

Ensure atlas paths are absolute and files exist:

```bash
# Test paths
ls -lh /path/to/atlas/ct.nii.gz
ls -lh /path/to/atlas/labels.nii.gz
```

### Label Count Mismatch

If output has different vertebrae count than configured:

```
[WARNING] Configured for 24 vertebrae but output contains 23 labels. Proceeding with available labels.
```

This is expected when input is missing vertebrae. Check:
- Input segmentation has expected label count
- Atlas has matching label count
- Atlas paths are correct

### Falling Back to Legacy Mode

If warmup mode fails, pipeline automatically falls back to legacy processing with warning:

```
[WARNING] Warmup mode requires fixed_ct_path, fixed_labels_path, moving_ct_path, and output_dir.
```

Ensure config.yaml has all required paths for warmup mode.

## Examples

### Example 1: Full Spine with Default Settings

```yaml
vertebrae:
  processing_mode: "warmup"
  target_count: 24
  fixed_reference_ct: "/data/atlas/BDMAP_00000031/ct.nii.gz"
  fixed_reference_labels: "/data/atlas/BDMAP_00000031/labels.nii.gz"
  warmup_config:
    cleanup: {}  # Uses defaults
    registration: {}
    relabel: {}
    finalize: {}
```

### Example 2: Thoracic Spine Only

```yaml
vertebrae:
  processing_mode: "warmup"
  target_count: 12
  fixed_reference_ct: "/data/atlas/thoracic_reference/ct.nii.gz"
  fixed_reference_labels: "/data/atlas/thoracic_reference/labels.nii.gz"
  warmup_config:
    finalize:
      target_labels: 12  # Match target_count
```

### Example 3: Aggressive Merging for Low Vertebrae

```yaml
vertebrae:
  processing_mode: "warmup"
  target_count: 24
  # ... atlas paths ...
  warmup_config:
    finalize:
      max_edit_label: 7  # Only edit cervical (C1-C7)
      low_label_aggressiveness: 1.5  # More aggressive merging
```

## API Usage (Python)

You can also use the warmup postprocessing directly in Python:

```python
from pathlib import Path
from utils.vertebrae_warmup_postprocessing import (
    process_vertebrae_warmup,
    CleanupConfig,
    RegistrationConfig,
    RelabelConfig,
    FinalizeConfig,
)
import nibabel as nib
import numpy as np

# Load input labels
input_labels = np.asanyarray(nib.load("input_labels.nii.gz").dataobj).astype(np.int16)

# Run warmup processing
final_labels, audit_trails = process_vertebrae_warmup(
    input_labels=input_labels,
    fixed_ct_path="/path/to/atlas/ct.nii.gz",
    fixed_labels_path="/path/to/atlas/labels.nii.gz",
    moving_ct_path="/path/to/input/ct.nii.gz",
    output_dir=Path("/path/to/output"),
    target_vertebrae_count=24,
    verbose=False,
)

# Access audit trails
print(audit_trails['split_components'])  # List of CSV rows
print(audit_trails['volume_merge_pairs'])
```

## Citation

If you use the vertebrae warmup postprocessing pipeline, please cite:
- ShapeKit: [ShapeKit GitHub](https://github.com/BodyMaps/ShapeKit)
- ANTsPy: [ANTsPy Documentation](https://antspy.readthedocs.io/)

## Support

For issues or questions:
1. Check the audit trail CSVs to understand processing decisions
2. Review configuration parameters for potential adjustments
3. Test with legacy mode as baseline
4. Check atlas path validity and label ranges
