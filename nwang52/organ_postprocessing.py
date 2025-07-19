import os
import nibabel as nib
import numpy as np
from pathlib import Path
import logging
import argparse
from scipy import ndimage
import multiprocessing
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
# Import class maps
from class_maps import available_class_maps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_organ_maps(seg_dir):
    """
    Dynamically build organ maps by scanning the segmentations directory

    Args:
        seg_dir: Path to the segmentations directory

    Returns:
        organ_label_map: Dictionary mapping organ name to label ID, format {organ_name: label_id}
        organ_processing_params: Dictionary of organ processing parameters
    """
    if not os.path.exists(seg_dir):
        logging.error(f"segmentations directory does not exist: {seg_dir}")
        return {}, {}

    # Get all .nii.gz files
    organ_files = [f[:-7] for f in os.listdir(seg_dir) if f.endswith('.nii.gz')]

    if not organ_files:
        logging.error(f"No .nii.gz files found in segmentations directory: {seg_dir}")
        return {}, {}

    # Build label map, starting from 1 in order
    organ_label_map = {name: idx + 1 for idx, name in enumerate(sorted(organ_files))}

    # Build processing parameter map, all organs use the same thresholds
    organ_processing_params = {
        name: {'min_size_ratio': 0.25, 'min_merge_ratio': 0.075}
        for name in organ_files
    }

    logging.info(f"Found {len(organ_files)} organs")
    for name, label in organ_label_map.items():
        logging.info(f"  {name}: {label}")

    return organ_label_map, organ_processing_params

def remove_small_components(organ_mask, combined_labels=None, current_label=None, min_size_ratio=0.05, min_merge_ratio=0.05, ORGAN_LABEL_MAP=None):
    """
    Remove small connected components in organ segmentation, merge larger noise regions

    Args:
        organ_mask: Binary mask of the organ
        combined_labels: Full label image
        current_label: Current organ label being processed
        min_size_ratio: Minimum volume ratio relative to the largest component
        min_merge_ratio: Minimum volume ratio to consider merging
        ORGAN_LABEL_MAP: Organ label map
    """
    if not np.any(organ_mask):
        return organ_mask, {}

    labeled_organ, num_features = ndimage.label(organ_mask)
    if num_features <= 1:
        return organ_mask, {}

    # Calculate the size of each connected component
    component_sizes = ndimage.sum(organ_mask, labeled_organ, range(1, num_features + 1))
    max_size = np.max(component_sizes)
    size_threshold = max_size * min_size_ratio
    min_merge_size = max_size * min_merge_ratio  # Dynamically calculate merge threshold

    # For spleen, dynamically set merge_size
    if min_merge_size == 'dynamic':
        min_merge_size = max_size * 0.1  # Use 10% of the largest component as merge threshold

    # Create new mask, initialize with the largest component
    max_component_idx = np.argmax(component_sizes) + 1
    cleaned_mask = (labeled_organ == max_component_idx)

    # Record regions to be merged
    merge_regions = {}

    # Process other components
    for i, size in enumerate(component_sizes, 1):
        if i == max_component_idx:
            continue

        component = (labeled_organ == i)

        # 1. If larger than threshold (25%), keep directly
        if size >= size_threshold:
            cleaned_mask |= component
        else:
            # If volume > min_merge_size, try to merge
            if size >= min_merge_size and combined_labels is not None and current_label is not None and ORGAN_LABEL_MAP is not None:
                # Get the border of this component
                dilated = ndimage.binary_dilation(component)
                border = dilated & ~component

                # Find neighboring labels on the border
                neighbor_labels = np.unique(combined_labels[border])
                neighbor_labels = neighbor_labels[(neighbor_labels > 0) & (neighbor_labels != current_label)]

                if len(neighbor_labels) > 0:
                    # Calculate contact area with each neighboring label
                    contact_areas = [(label, np.sum(border & (combined_labels == label)))
                                   for label in neighbor_labels]
                    target_label = max(contact_areas, key=lambda x: x[1])[0]
                    merge_regions[int(target_label)] = merge_regions.get(int(target_label), [])
                    merge_regions[int(target_label)].append(component)

                    # Get current and target organ names
                    current_organ = next((name for name, id in ORGAN_LABEL_MAP.items() if id == current_label), f"Unknown-{current_label}") if ORGAN_LABEL_MAP and current_label else "Unknown"
                    target_organ = next((name for name, id in ORGAN_LABEL_MAP.items() if id == target_label), f"Unknown-{target_label}")
                    logging.info(f"Merging connected component of {current_organ} with volume {size} into {target_organ}")
            else:
                # If volume < min_merge_size, remove directly
                current_organ = next((name for name, id in ORGAN_LABEL_MAP.items() if id == current_label), f"Unknown-{current_label}") if ORGAN_LABEL_MAP and current_label else "Unknown"
                logging.info(f"Removing connected component of {current_organ} with volume {size} (below threshold {size_threshold} and {min_merge_size})")

    return cleaned_mask, merge_regions

def process_femur(combined_labels, bladder_seg, ORGAN_LABEL_MAP):
    """Handle mixing of left and right femur, filter noise based on bladder location"""
    # Check if required organs exist
    if 'femur_left' not in ORGAN_LABEL_MAP or 'femur_right' not in ORGAN_LABEL_MAP:
        logging.warning("Left or right femur label not found, skipping femur processing")
        return combined_labels
        
    # Get all femur regions (regardless of side)
    femur_mask = (combined_labels == ORGAN_LABEL_MAP['femur_left']) | \
                 (combined_labels == ORGAN_LABEL_MAP['femur_right'])
    
    if not np.any(femur_mask):
        return combined_labels

    # 1. Get reference kidney position
    kidney_left_exists = 'kidney_left' in ORGAN_LABEL_MAP
    kidney_right_exists = 'kidney_right' in ORGAN_LABEL_MAP
    
    kidney_left_mask = (combined_labels == ORGAN_LABEL_MAP['kidney_left']) if kidney_left_exists else np.zeros_like(combined_labels, dtype=bool)
    kidney_right_mask = (combined_labels == ORGAN_LABEL_MAP['kidney_right']) if kidney_right_exists else np.zeros_like(combined_labels, dtype=bool)
    
    use_right_kidney = False
    reference_x = 0
    
    if np.any(kidney_left_mask):
        reference_x = np.mean(np.where(kidney_left_mask)[0])
        logging.info(f"Using left kidney as reference, x={reference_x:.2f}")
    elif np.any(kidney_right_mask):
        reference_x = np.mean(np.where(kidney_right_mask)[0])
        use_right_kidney = True
        logging.info(f"Using right kidney as reference, x={reference_x:.2f}")
    else:
        logging.warning("No left or right kidney found, using image midline for femur processing")
        reference_x = femur_mask.shape[0] // 2

    # 2. Split femur regions into two groups by midline
    midline = femur_mask.shape[0] // 2
    new_labels = combined_labels.copy()
    new_labels[femur_mask] = 0  # Clear all femur labels
    
    # Connected component analysis for all femur regions
    labeled_femur, num_features = ndimage.label(femur_mask)
    
    # Split components into left and right groups
    left_side_components = []
    right_side_components = []
    
    for i in range(1, num_features + 1):
        component = labeled_femur == i
        centroid = ndimage.center_of_mass(component)
        if centroid[0] > midline:
            right_side_components.append(component)
        else:
            left_side_components.append(component)
    
    # 3. Calculate mean distance to reference kidney for each group
    def calc_group_distance(components):
        if not components:
            return float('inf')
        centroids = [ndimage.center_of_mass(comp)[0] for comp in components]
        return np.mean([abs(x - reference_x) for x in centroids])
    
    left_group_dist = calc_group_distance(left_side_components)
    right_group_dist = calc_group_distance(right_side_components)
    
    # 4. Assign labels based on distance
    if not use_right_kidney:  # Using left kidney as reference
        if left_group_dist < right_group_dist:
            for comp in left_side_components:
                new_labels[comp] = ORGAN_LABEL_MAP['femur_left']
            for comp in right_side_components:
                new_labels[comp] = ORGAN_LABEL_MAP['femur_right']
        else:
            for comp in left_side_components:
                new_labels[comp] = ORGAN_LABEL_MAP['femur_right']
            for comp in right_side_components:
                new_labels[comp] = ORGAN_LABEL_MAP['femur_left']
    else:  # Using right kidney as reference
        if left_group_dist < right_group_dist:
            for comp in left_side_components:
                new_labels[comp] = ORGAN_LABEL_MAP['femur_right']
            for comp in right_side_components:
                new_labels[comp] = ORGAN_LABEL_MAP['femur_left']
        else:
            for comp in left_side_components:
                new_labels[comp] = ORGAN_LABEL_MAP['femur_left']
            for comp in right_side_components:
                new_labels[comp] = ORGAN_LABEL_MAP['femur_right']

    # 5. Filter femur components by z-axis distance to bladder
    if np.any(bladder_seg):
        bladder_centroid = ndimage.center_of_mass(bladder_seg)
        bladder_z = bladder_centroid[2]
        logging.info(f"Bladder centroid z-coordinate: {bladder_z:.2f}")
        
        # Process left femur
        left_femur_mask = (new_labels == ORGAN_LABEL_MAP['femur_left'])
        labeled_left_femur, num_left = ndimage.label(left_femur_mask)
        if num_left > 0:
            # Calculate z-axis distances for all left femur components
            left_z_distances = []
            for i in range(1, num_left + 1):
                comp = labeled_left_femur == i
                comp_centroid = ndimage.center_of_mass(comp)
                z_dist = abs(comp_centroid[2] - bladder_z)
                left_z_distances.append((i, z_dist))
            
            # Calculate minimum z-distance
            min_left_z_dist = min(dist for _, dist in left_z_distances)
            threshold = min_left_z_dist * 10
            
            # Remove components with excessive distance
            for i, dist in left_z_distances:
                if dist > threshold:
                    comp = (labeled_left_femur == i)
                    new_labels[comp] = 0
                    logging.info(f"Removed left femur component {i}, z-distance: {dist:.2f} > threshold: {threshold:.2f}")
        
        # Process right femur
        right_femur_mask = (new_labels == ORGAN_LABEL_MAP['femur_right'])
        labeled_right_femur, num_right = ndimage.label(right_femur_mask)
        if num_right > 0:
            # Calculate z-axis distances for all right femur components
            right_z_distances = []
            for i in range(1, num_right + 1):
                comp = labeled_right_femur == i
                comp_centroid = ndimage.center_of_mass(comp)
                z_dist = abs(comp_centroid[2] - bladder_z)
                right_z_distances.append((i, z_dist))
            
            # Calculate minimum z-distance
            min_right_z_dist = min(dist for _, dist in right_z_distances)
            threshold = min_right_z_dist * 10
            
            # Remove components with excessive distance
            for i, dist in right_z_distances:
                if dist > threshold:
                    comp = (labeled_right_femur == i)
                    new_labels[comp] = 0
                    logging.info(f"Removed right femur component {i}, z-distance: {dist:.2f} > threshold: {threshold:.2f}")
    
    return new_labels

def process_prostate(combined_labels, femur_processed_labels, ORGAN_LABEL_MAP):
    """Process prostate based on femur position"""
    # Get x-axis range of left and right femur
    femur_left_mask = (femur_processed_labels == ORGAN_LABEL_MAP['femur_left'])
    femur_right_mask = (femur_processed_labels == ORGAN_LABEL_MAP['femur_right'])
    
    if not (np.any(femur_left_mask) and np.any(femur_right_mask)):
        logging.warning("Incomplete left/right femur info, skipping prostate processing")
        return combined_labels

    # Calculate x center of left and right femur
    left_x_indices = np.where(femur_left_mask)[0]
    right_x_indices = np.where(femur_right_mask)[0]
    femur_center_x = (np.mean(left_x_indices) + np.mean(right_x_indices)) / 2
    logging.info(f"Femur center x: {femur_center_x}")

    # Get prostate region
    prostate_mask = (combined_labels == ORGAN_LABEL_MAP['prostate'])
    if not np.any(prostate_mask):
        return combined_labels

    # Analyze prostate connected components
    labeled_prostate, num_features = ndimage.label(prostate_mask)
    new_labels = combined_labels.copy()
    new_labels[prostate_mask] = 0  # Clear original prostate label

    if num_features > 0:
        # Calculate x distance from each component to femur center
        distances = []
        for i in range(1, num_features + 1):
            component = (labeled_prostate == i)
            x_indices = np.where(component)[0]
            component_center_x = np.mean(x_indices)
            distance = abs(component_center_x - femur_center_x)
            distances.append({
                'idx': i,
                'distance': distance,
                'mask': component,
                'center_x': component_center_x
            })
            logging.info(f"Prostate component {i} center x: {component_center_x:.2f}, distance to femur center: {distance:.2f}")

        # Keep only the closest component
        if distances:
            closest = min(distances, key=lambda x: x['distance'])
            new_labels[closest['mask']] = ORGAN_LABEL_MAP['prostate']
            logging.info(f"Keep prostate component closest to femur center, distance: {closest['distance']:.2f}")

    return new_labels

def fix_lung_overlap(lung_left_seg, lung_right_seg, seg_dir):
    """Fix overlap between left and right lung, determine midline based on aorta position"""
    # Merge left and right lung
    combined_lung = lung_left_seg | lung_right_seg

    # If no lung segmentation or only one side, return directly
    if not np.any(combined_lung) or not np.any(lung_left_seg) or not np.any(lung_right_seg):
        return lung_left_seg, lung_right_seg

    # Load aorta segmentation and calculate its centroid as midline
    aorta_path = os.path.join(seg_dir, "aorta.nii.gz")
    if os.path.exists(aorta_path):
        aorta_nii = nib.load(aorta_path)
        aorta_mask = aorta_nii.get_fdata() > 0
        if np.any(aorta_mask):
            aorta_centroid = ndimage.center_of_mass(aorta_mask)
            midline = aorta_centroid[0]
            logging.info(f"Using aorta centroid x {midline:.2f} as midline")
        else:
            midline = combined_lung.shape[0] // 2
            logging.warning("Aorta mask is empty, using image midline as default")
    else:
        midline = combined_lung.shape[0] // 2
        logging.warning("Aorta segmentation file not found, using image midline as default")
    
    # Calculate centroids of left and right lung
    left_centroid = ndimage.center_of_mass(lung_left_seg)
    right_centroid = ndimage.center_of_mass(lung_right_seg)
    
    # Record centroids' x-coordinates
    left_centroid_x = left_centroid[0]
    right_centroid_x = right_centroid[0]
    logging.info(f"Left lung centroid x: {left_centroid_x:.2f}, Right lung centroid x: {right_centroid_x:.2f}")
    
    # Check if lungs are swapped based on centroids' x-coordinates
    swapped = False
    if left_centroid_x > right_centroid_x:
        logging.info("Detected swapped lung labels, adjusting based on centroids' position")
        swapped = True
        # Swap left and right lung segmentation
        lung_left_seg, lung_right_seg = lung_right_seg, lung_left_seg
        # Update centroids
        left_centroid_x, right_centroid_x = right_centroid_x, left_centroid_x
    
    # Create new masks for left and right lung
    new_left_mask = lung_left_seg.copy()
    new_right_mask = lung_right_seg.copy()
    
    # Detect overlap
    overlap = new_left_mask & new_right_mask
    
    if np.any(overlap):
        logging.info(f"Detected overlap between lungs, volume: {np.sum(overlap)}")
        
        # Get all x-coordinates of the overlap
        overlap_x_coords = np.unique(np.where(overlap)[0])
        
        # Assign overlap region based on midline and centroids
        if (left_centroid_x < midline and right_centroid_x < midline) or \
           (left_centroid_x > midline and right_centroid_x > midline):
            logging.info("Both lung centroids on the same side of midline, using midline for separation")
            for x in overlap_x_coords:
                if x < midline:
                    # Midline left side belongs to left lung
                    new_right_mask[x, :, :] &= ~overlap[x, :, :]
                else:
                    # Midline right side belongs to right lung
                    new_left_mask[x, :, :] &= ~overlap[x, :, :]
        else:
            # Use the midpoint of the centroids as the boundary
            boundary = (left_centroid_x + right_centroid_x) / 2
            logging.info(f"Using midpoint of lung centroids {boundary:.2f} as boundary")
            
            for x in overlap_x_coords:
                if x < boundary:
                    # Boundary left side belongs to left lung
                    new_right_mask[x, :, :] &= ~overlap[x, :, :]
                else:
                    # Boundary right side belongs to right lung
                    new_left_mask[x, :, :] &= ~overlap[x, :, :]
    
    # Handle unassigned regions if any
    unassigned = combined_lung & ~(new_left_mask | new_right_mask)
    
    if np.any(unassigned):
        logging.info(f"Detected unassigned lung regions, volume: {np.sum(unassigned)}")
        
        # Get all x-coordinates of the unassigned region
        unassigned_x_coords = np.unique(np.where(unassigned)[0])
        
        # Assign unassigned region using the same logic as overlap
        if (left_centroid_x < midline and right_centroid_x < midline) or \
           (left_centroid_x > midline and right_centroid_x > midline):
            for x in unassigned_x_coords:
                if x < midline:
                    # Midline left side belongs to left lung
                    new_left_mask[x, :, :] |= unassigned[x, :, :]
                else:
                    # Midline right side belongs to right lung
                    new_right_mask[x, :, :] |= unassigned[x, :, :]
        else:
            boundary = (left_centroid_x + right_centroid_x) / 2
            for x in unassigned_x_coords:
                if x < boundary:
                    # Boundary left side belongs to left lung
                    new_left_mask[x, :, :] |= unassigned[x, :, :]
                else:
                    # Boundary right side belongs to right lung
                    new_right_mask[x, :, :] |= unassigned[x, :, :]
    
    # If initially detected swapped lungs, swap results back
    if swapped:
        logging.info("Swapping results back due to initial label swap")
        new_left_mask, new_right_mask = new_right_mask, new_left_mask
    
    return new_left_mask, new_right_mask

def fix_liver_segmentation(case_dir, output_dir=None, class_map_name="all"):
    """Fix errors in liver segmentation using other organs as spatial reference
    
    Args:
        case_dir: Path to the case directory
        output_dir: Path to the output directory. If None, saves in a subdirectory of case_dir
        class_map_name: Name of the class map to use for processing (default: "all")
    
    Returns:
        bool: True if processing is successful, False otherwise
    """
    # Get the specified class map
    if class_map_name not in available_class_maps:
        logging.error(f"Unknown class map: {class_map_name}. Available: {list(available_class_maps.keys())}")
        return False
    
    class_map = available_class_maps[class_map_name]
    # Create reverse mapping from organ name to class ID
    PROCESS_ORGAN_MAP = {organ_name: class_id for class_id, organ_name in class_map.items()}
    
    logging.info(f"Using class map '{class_map_name}' with {len(PROCESS_ORGAN_MAP)} organs to process")
    
    # Determine the output directory
    if output_dir is None:
        # Original behavior: save in after_processing subdirectory
        after_processing_dir = os.path.join(case_dir, "after_processing")
    else:
        # New behavior: save in the corresponding subdirectory of the output directory
        case_name = os.path.basename(os.path.normpath(case_dir))
        after_processing_dir = os.path.join(output_dir, case_name)
    
    # Set log file for this case
    os.makedirs(after_processing_dir, exist_ok=True)
    log_file = os.path.join(after_processing_dir, "postprocessing.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"Start processing case: {case_dir}")
    logging.info(f"Output will be saved to: {after_processing_dir}")
    
    # Path definitions
    seg_dir = os.path.join(case_dir, "segmentations")
    
    # Dynamically build organ maps (load ALL organs from segmentations)
    ORGAN_LABEL_MAP, ORGAN_PROCESSING_PARAMS = build_organ_maps(seg_dir)
    
    # Check if liver exists in both maps
    if 'liver' not in ORGAN_LABEL_MAP:
        logging.error("Missing required organ: liver in segmentations")
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        return False
    
    if 'liver' not in PROCESS_ORGAN_MAP:
        logging.error("Missing required organ: liver in class map")
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        return False
    
    # Create output directory for processed files
    os.makedirs(after_processing_dir, exist_ok=True)
    
    # Get image information and data type from the first found organ segmentation
    sample_nii = None
    for name in ORGAN_LABEL_MAP.keys():
        organ_path = os.path.join(seg_dir, f"{name}.nii.gz")
        if os.path.exists(organ_path):
            sample_nii = nib.load(organ_path)
            break
    
    if sample_nii is None:
        logging.error("No organ segmentation files found")
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        return False
        
    # Get original data type and affine transformation matrix
    original_dtype = sample_nii.get_data_dtype()
    affine_matrix = sample_nii.affine
        
    # Create empty combined_labels
    image_shape = sample_nii.shape
    combined_labels = np.zeros(image_shape, dtype=original_dtype)
    
    # Load segmentations for ALL organs and merge into combined_labels
    for name, label_id in ORGAN_LABEL_MAP.items():
        organ_path = os.path.join(seg_dir, f"{name}.nii.gz")
        if os.path.exists(organ_path):
            organ_nii = nib.load(organ_path)
            organ_mask = organ_nii.get_fdata() > 0
            combined_labels[organ_mask] = label_id
    
    # Create NIfTI object for saving
    combined_labels_nii = nib.Nifti1Image(combined_labels, affine_matrix)
    combined_labels_nii.set_data_dtype(original_dtype)
    
    # Create a copy for further processing
    new_combined_labels = combined_labels.copy()
    
    # Fix overlap in lung segmentation - check if lung files exist
    lung_left_path = os.path.join(seg_dir, "lung_left.nii.gz")
    lung_right_path = os.path.join(seg_dir, "lung_right.nii.gz")
    
    if os.path.exists(lung_left_path) and os.path.exists(lung_right_path):
        lung_left_seg = nib.load(lung_left_path).get_fdata() > 0
        lung_right_seg = nib.load(lung_right_path).get_fdata() > 0
        
        # Fix lung overlap issue
        fixed_left_lung, fixed_right_lung = fix_lung_overlap(lung_left_seg, lung_right_seg, seg_dir)
        
        # Update lung labels - using new_combined_labels
        if 'lung_left' in ORGAN_LABEL_MAP:
            new_combined_labels[new_combined_labels == ORGAN_LABEL_MAP['lung_left']] = 0
            new_combined_labels[fixed_left_lung] = ORGAN_LABEL_MAP['lung_left']
        
        if 'lung_right' in ORGAN_LABEL_MAP:
            new_combined_labels[new_combined_labels == ORGAN_LABEL_MAP['lung_right']] = 0
            new_combined_labels[fixed_right_lung] = ORGAN_LABEL_MAP['lung_right']
        
        # Use corrected lungs to calculate data
        lung_left_x_indices = np.where(fixed_left_lung)[0]
        lung_right_x_indices = np.where(fixed_right_lung)[0]
    else:
        logging.warning("Lung segmentation files not found, skipping lung overlap correction")
        fixed_left_lung = np.zeros_like(combined_labels, dtype=bool)
        fixed_right_lung = np.zeros_like(combined_labels, dtype=bool)
        lung_left_x_indices = np.array([])
        lung_right_x_indices = np.array([])
    
    # Calculate total volume of left and right lungs
    lung_total_volume = np.sum(fixed_left_lung) + np.sum(fixed_right_lung)
    logging.info(f"Total lung volume: {lung_total_volume}")
    
    # Check if lung regions exist for x-axis constraint
    has_lung_for_x_constraint = (len(lung_left_x_indices) > 0 or len(lung_right_x_indices) > 0)
    
    # Calculate x-axis range of lungs for liver filtering
    lung_x_min = min(np.min(lung_left_x_indices) if len(lung_left_x_indices) > 0 else float('inf'),
                     np.min(lung_right_x_indices) if len(lung_right_x_indices) > 0 else float('inf'))
    lung_x_max = max(np.max(lung_left_x_indices) if len(lung_left_x_indices) > 0 else 0,
                     np.max(lung_right_x_indices) if len(lung_right_x_indices) > 0 else 0)
    
    # If no lungs found, use 25% to 75% of image width as default range
    if lung_x_min == float('inf'):
        lung_x_min = combined_labels.shape[0] // 4
    if lung_x_max == 0:
        lung_x_max = combined_labels.shape[0] * 3 // 4
        
    # Record lung x-axis range information
    if has_lung_for_x_constraint:
        logging.info(f"Lung x-axis range: [{lung_x_min}, {lung_x_max}]")
    else:
        logging.warning("No lungs found, will not apply x-axis constraint to liver")

    # Load bladder segmentation for femur processing
    bladder_exists = 'bladder' in ORGAN_LABEL_MAP
    if bladder_exists:
        bladder_path = os.path.join(seg_dir, "bladder.nii.gz")
        bladder_seg = nib.load(bladder_path).get_fdata() > 0
    else:
        bladder_seg = np.zeros_like(combined_labels, dtype=bool)
        logging.warning("Bladder segmentation not found, using empty mask")
    
    # Calculate z-axis range of right lung for liver filtering
    has_lung_right = 'lung_right' in ORGAN_LABEL_MAP and np.any(fixed_right_lung)
    lung_right_center = None
    lung_right_volume = 0
    if has_lung_right:
        # Calculate maximum connected component volume of right lung
        labeled_lung_right, num_features = ndimage.label(fixed_right_lung)
        if num_features > 0:
            lung_right_component_sizes = ndimage.sum(fixed_right_lung, labeled_lung_right, range(1, num_features + 1))
            lung_right_volume = np.max(lung_right_component_sizes)
            lung_right_z_indices = np.where(fixed_right_lung)[2]
            lung_right_center = (np.min(lung_right_z_indices) + np.max(lung_right_z_indices)) // 2
            logging.info(f"Right lung center position: {lung_right_center}, maximum component volume: {lung_right_volume}")
    else:
        logging.warning("Right lung segmentation not found or empty, will not apply z-axis constraint to liver")
    
    # Get maximum connected component volume and centroid of bladder
    has_bladder = 'bladder' in ORGAN_LABEL_MAP
    bladder_center = None
    bladder_volume = 0
    if has_bladder:
        bladder_seg = (combined_labels == ORGAN_LABEL_MAP['bladder'])
        if np.any(bladder_seg):
            # Calculate maximum connected component volume of bladder
            labeled_bladder, num_features = ndimage.label(bladder_seg)
            if num_features > 0:
                bladder_component_sizes = ndimage.sum(bladder_seg, labeled_bladder, range(1, num_features + 1))
                bladder_volume = np.max(bladder_component_sizes)
                bladder_z_indices = np.where(bladder_seg)[2]
                bladder_center = (np.min(bladder_z_indices) + np.max(bladder_z_indices)) // 2
                logging.info(f"Bladder center position: {bladder_center}, maximum component volume: {bladder_volume}")
        else:
            logging.warning("Bladder segmentation is empty")
    else:
        logging.warning("Bladder label not found")

    # Get liver regions
    liver_label = ORGAN_LABEL_MAP['liver']
    liver_regions = (new_combined_labels == liver_label)
    original_liver_volume = np.sum(liver_regions)
    
    # Process liver regions using connected component analysis
    labeled_liver, num_features = ndimage.label(liver_regions)
    
    # Add warning for excessive fragmentation and filter components
    if num_features > 50:
        logging.warning(f"Liver has {num_features} connected components, which indicates severe fragmentation.")
        logging.info(f"Applying fragmentation filter: keeping only the largest 20 components")
        
        # Calculate sizes of all components
        liver_component_sizes = ndimage.sum(liver_regions, labeled_liver, range(1, num_features + 1))
        
        # Get indices of the 20 largest components
        largest_component_indices = np.argsort(liver_component_sizes)[-20:]  # Get indices of 20 largest
        largest_component_labels = largest_component_indices + 1  # Convert to 1-based labeling
        
        # Create filtered liver mask with only the 20 largest components
        filtered_liver_mask = np.zeros_like(liver_regions)
        for label in largest_component_labels:
            component_mask = (labeled_liver == label)
            filtered_liver_mask |= component_mask
        
        # Log size information for kept components
        kept_sizes = liver_component_sizes[largest_component_indices]
        total_kept_volume = np.sum(kept_sizes)
        removed_volume = original_liver_volume - total_kept_volume
        
        logging.info(f"Kept {len(largest_component_labels)} largest components:")
        logging.info(f"  Largest component size: {np.max(kept_sizes)}")
        logging.info(f"  Smallest kept component size: {np.min(kept_sizes)}")
        logging.info(f"  Total kept volume: {total_kept_volume} ({total_kept_volume/original_liver_volume*100:.1f}%)")
        logging.info(f"  Removed volume: {removed_volume} ({removed_volume/original_liver_volume*100:.1f}%)")
        
        # Update liver regions and relabel
        liver_regions = filtered_liver_mask
        labeled_liver, num_features = ndimage.label(liver_regions)
        logging.info(f"After fragmentation filtering: {num_features} components remaining")
    
    # Calculate liver's maximum connected component volume and check if lung and bladder volumes meet criteria
    if num_features > 0:
        liver_component_sizes = ndimage.sum(liver_regions, labeled_liver, range(1, num_features + 1))
        max_liver_size = np.max(liver_component_sizes)
        logging.info(f"Liver has {num_features} connected components, maximum component volume: {max_liver_size}")
        
        # Check if total lung volume is large enough
        lung_volume_threshold = 10000  # Fixed threshold of 10000
        has_sufficient_lung_volume = lung_total_volume >= lung_volume_threshold
        
        # Check if right lung and bladder volumes are large enough (for z-axis constraint)
        lung_right_volume_threshold = 10000  # Fixed threshold of 10000
        bladder_volume_threshold = 10000     # Fixed threshold of 10000
        
        has_sufficient_z_reference = (lung_right_volume >= lung_right_volume_threshold and 
                                    bladder_volume >= bladder_volume_threshold)
        
        if not has_sufficient_lung_volume:
            logging.warning(f"Total lung volume ({lung_total_volume}) less than threshold ({lung_volume_threshold}), will not apply x-axis constraint to liver")
        if not has_sufficient_z_reference:
            logging.warning(f"Right lung volume ({lung_right_volume}) or bladder volume ({bladder_volume}) does not meet threshold requirement (10000), will not apply z-axis constraint to liver")
    else:
        has_sufficient_lung_volume = False
        has_sufficient_z_reference = False
        logging.warning("No liver connected components found after processing")
    
    # Determine whether to use x-axis and z-axis constraints
    apply_x_constraint = has_lung_for_x_constraint and has_sufficient_lung_volume
    apply_z_constraint = (bladder_center is not None and 
                         lung_right_center is not None and 
                         has_sufficient_z_reference)
    
    liver_mask = liver_regions.copy()
    
    # Analyze range of each liver connected component with progress logging
    components_to_remove = []
    components_to_keep = []
    
    for i in range(1, num_features + 1):
        # Log progress for large numbers of components
        if num_features > 20 and i % 10 == 0:
            logging.info(f"Processing liver component {i}/{num_features} ({(i/num_features)*100:.1f}%)")
        
        component = labeled_liver == i
        z_indices = np.where(component)[2]
        x_indices = np.where(component)[0]
        
        if len(z_indices) == 0 or len(x_indices) == 0:
            components_to_remove.append(i)
            continue
            
        z_min, z_max = np.min(z_indices), np.max(z_indices)
        
        # Calculate centroid of connected component
        z_center = (z_min + z_max) / 2
        x_center = np.mean(x_indices)
        
        # Initialize judgment conditions
        should_remove = False
        removal_reason = ""
        
        # Check if z-axis is within range (if constraint conditions are met)
        if apply_z_constraint:
            z_in_range = (min(bladder_center, lung_right_center) <= z_center <= max(bladder_center, lung_right_center))
            if not z_in_range:
                should_remove = True
                removal_reason = f"z-axis center {z_center:.2f} not between bladder center {bladder_center} and right lung center {lung_right_center}"
        
        # Only check x-axis range if applying x-axis constraint and not already marked for deletion
        if apply_x_constraint and not should_remove:
            x_tolerance = (lung_x_max - lung_x_min) * 0.1  # 10% tolerance
            x_in_range = (lung_x_min - x_tolerance <= x_center <= lung_x_max + x_tolerance)
            if not x_in_range:
                should_remove = True
                removal_reason = f"x-axis center {x_center:.2f} not within lung x-axis range [{lung_x_min - x_tolerance:.2f}, {lung_x_max + x_tolerance:.2f}]"
        
        if should_remove:
            components_to_remove.append(i)
            if num_features <= 20:  # Only log individual components for smaller numbers
                logging.info(f"Removing liver connected component {i}, reason: {removal_reason}")
        else:
            components_to_keep.append(i)
            if num_features <= 20:  # Only log individual components for smaller numbers
                logging.info(f"Keeping liver connected component {i}, center position: x={x_center:.2f}, z={z_center:.2f}")

    # Batch update liver mask
    for i in components_to_remove:
        component = labeled_liver == i
        liver_mask[component] = False
    
    # Log summary
    logging.info(f"Liver component processing complete: kept {len(components_to_keep)}, removed {len(components_to_remove)} components")
    
    # Create new combined_labels, preserving original labels
    new_combined_labels[liver_regions] = 0
    new_combined_labels[liver_mask] = liver_label
    
    # Remove noise from each organ - BUT ONLY FOR ORGANS IN CLASS MAP
    merge_updates = {}  # Record all regions to be merged
    
    for name, label_id in ORGAN_LABEL_MAP.items():
        if label_id == liver_label:  # Skip already processed liver
            continue
        
        # Only process organs that are in the class map
        if name not in PROCESS_ORGAN_MAP:
            logging.info(f"Skipping noise removal for {name} (not in class map)")
            continue
        
        # Get organ-specific parameters
        organ_params = ORGAN_PROCESSING_PARAMS[name]
        
        organ_mask = (new_combined_labels == label_id)
        if np.any(organ_mask):
            logging.info(f"Processing noise removal for {name}")
            # Remove small connected components
            cleaned_mask, merge_regions = remove_small_components(
                organ_mask, 
                combined_labels=new_combined_labels,
                current_label=label_id,
                min_size_ratio=organ_params['min_size_ratio'],
                min_merge_ratio=organ_params['min_merge_ratio'],
                ORGAN_LABEL_MAP=ORGAN_LABEL_MAP
            )
            
            # Update labels
            new_combined_labels[organ_mask] = 0  # Clear original region
            new_combined_labels[cleaned_mask] = label_id  # Add cleaned region
            
            # Record regions to be merged
            for target_label, regions in merge_regions.items():
                if target_label not in merge_updates:
                    merge_updates[target_label] = []
                merge_updates[target_label].extend(regions)
    
    # Process all regions to be merged
    for target_label, regions in merge_updates.items():
        for target_label, regions in merge_updates.items():
            for region in regions:
                new_combined_labels[region] = target_label
                logging.info(f"Completed region merging to label {target_label}")
    
    # Calculate corrected volume
    corrected_liver_volume = np.sum(new_combined_labels == liver_label)
    logging.info(f"Liver segmentation correction complete:")
    logging.info(f"  Original volume: {original_liver_volume}")
    logging.info(f"  Corrected volume: {corrected_liver_volume} (reduced by {(original_liver_volume - corrected_liver_volume) / original_liver_volume * 100:.2f}%)")
    
    # After noise removal and region merging for each organ is completed
    
    # Add femur processing - only if both femurs are in class map
    if ('femur_left' in ORGAN_LABEL_MAP and 'femur_right' in ORGAN_LABEL_MAP and
        'femur_left' in PROCESS_ORGAN_MAP and 'femur_right' in PROCESS_ORGAN_MAP):
        logging.info("Starting femur processing...")
        new_combined_labels = process_femur(new_combined_labels, bladder_seg, ORGAN_LABEL_MAP)
        logging.info("Femur processing complete")
    else:
        logging.info("Skipping femur processing (not in class map or not found)")

    # Special processing for pancreas - only if in class map
    if 'pancreas' in ORGAN_LABEL_MAP and 'pancreas' in PROCESS_ORGAN_MAP:
        logging.info("Starting special processing for pancreas...")
        new_combined_labels = process_pancreas(new_combined_labels, ORGAN_LABEL_MAP)
    else:
        logging.info("Skipping pancreas processing (not in class map or not found)")
    
    # Save updated combined labels
    corrected_combined_nii = nib.Nifti1Image(new_combined_labels.astype(original_dtype), 
                                           combined_labels_nii.affine,
                                           combined_labels_nii.header.copy())
    corrected_combined_nii.set_data_dtype(original_dtype)
    corrected_combined_path = os.path.join(after_processing_dir, "combined_labels.nii.gz")
    nib.save(corrected_combined_nii, corrected_combined_path)
    
    # Create segmentations directory
    after_seg_dir = os.path.join(after_processing_dir, "segmentations")
    os.makedirs(after_seg_dir, exist_ok=True)
    
    # Iterate through ALL organ files in the original segmentations directory (preserve all files)
    for organ_file in os.listdir(seg_dir):
        if not organ_file.endswith('.nii.gz'):
            continue
            
        organ_name = organ_file[:-7]  # Remove .nii.gz suffix
        if organ_name in ORGAN_LABEL_MAP:
            # Extract organ mask from combined_labels
            organ_mask = (new_combined_labels == ORGAN_LABEL_MAP[organ_name])
            # Save all organs (processed and unprocessed)
            organ_nii = nib.Nifti1Image((organ_mask > 0).astype(np.float32), 
                                      combined_labels_nii.affine,
                                      combined_labels_nii.header.copy())
            organ_path = os.path.join(after_seg_dir, organ_file)
            nib.save(organ_nii, organ_path)
            
            # Log whether this organ was processed or not
            if organ_name in PROCESS_ORGAN_MAP:
                logging.info(f"Saved processed organ segmentation: {organ_file}")
            else:
                logging.info(f"Saved unprocessed organ segmentation: {organ_file}")
        else:
            logging.warning(f"Organ {organ_name} not found in label mapping")
    
    logging.info(f"Processed files have been saved to: {output_dir}")
    logging.info(f"Individual organ segmentation files have been saved to: {after_seg_dir}")
    logging.info(f"Processed organs using class map '{class_map_name}': {list(PROCESS_ORGAN_MAP.keys())}")
    
    # Remove file handler after processing
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()
    return True

def process_pancreas(combined_labels, ORGAN_LABEL_MAP):
    """Special processing for pancreas, merging connected components based on their centroids' position relative to lungs and stomach"""
    # Check if required organs exist
    if 'pancreas' not in ORGAN_LABEL_MAP:
        logging.warning("Pancreas label not found, skipping pancreas processing")
        return combined_labels
    
    # Check if lungs and stomach exist
    has_lungs = 'lung_left' in ORGAN_LABEL_MAP and 'lung_right' in ORGAN_LABEL_MAP
    has_stomach = 'stomach' in ORGAN_LABEL_MAP
    
    if not (has_lungs and has_stomach):
        logging.warning("Missing lungs or stomach labels, skipping advanced pancreas processing")
        return combined_labels
    
    # Get all pancreas related organ labels
    pancreas_labels = [label for name, label in ORGAN_LABEL_MAP.items() if 'pancreas' in name]
    logging.info(f"Found {len(pancreas_labels)} pancreas related labels: {pancreas_labels}")
    
    # Get main pancreas region
    pancreas_mask = (combined_labels == ORGAN_LABEL_MAP['pancreas'])
    
    # Get lung, stomach and pancreas regions
    lung_left_mask = (combined_labels == ORGAN_LABEL_MAP['lung_left'])
    lung_right_mask = (combined_labels == ORGAN_LABEL_MAP['lung_right'])
    stomach_mask = (combined_labels == ORGAN_LABEL_MAP['stomach'])
    
    # Check if organ masks are empty
    if not (np.any(lung_left_mask) and np.any(lung_right_mask) and np.any(stomach_mask) and np.any(pancreas_mask)):
        logging.warning("Lung, stomach or pancreas mask is empty, skipping advanced pancreas processing")
        return combined_labels
    
    # Analyze lung connected components
    labeled_lung_left, num_lung_left = ndimage.label(lung_left_mask)
    labeled_lung_right, num_lung_right = ndimage.label(lung_right_mask)
    
    # Get largest lung connected components
    if num_lung_left > 0:
        lung_left_sizes = ndimage.sum(lung_left_mask, labeled_lung_left, range(1, num_lung_left + 1))
        lung_left_largest = labeled_lung_left == (np.argmax(lung_left_sizes) + 1)
        lung_left_size = np.max(lung_left_sizes)
    else:
        lung_left_largest = np.zeros_like(lung_left_mask)
        lung_left_size = 0
    
    if num_lung_right > 0:
        lung_right_sizes = ndimage.sum(lung_right_mask, labeled_lung_right, range(1, num_lung_right + 1))
        lung_right_largest = labeled_lung_right == (np.argmax(lung_right_sizes) + 1)
        lung_right_size = np.max(lung_right_sizes)
    else:
        lung_right_largest = np.zeros_like(lung_right_mask)
        lung_right_size = 0
    
    # Analyze stomach connected components
    labeled_stomach, num_stomach = ndimage.label(stomach_mask)
    
    if num_stomach > 0:
        stomach_sizes = ndimage.sum(stomach_mask, labeled_stomach, range(1, num_stomach + 1))
        stomach_largest = labeled_stomach == (np.argmax(stomach_sizes) + 1)
        stomach_size = np.max(stomach_sizes)
    else:
        stomach_largest = np.zeros_like(stomach_mask)
        stomach_size = 0
    
    # Check if lung and stomach largest connected components are large enough
    min_organ_size = 2000
    if lung_left_size < min_organ_size or lung_right_size < min_organ_size or stomach_size < min_organ_size:
        logging.warning(f"Lung or stomach largest connected component less than threshold ({min_organ_size}), skipping advanced pancreas processing")
        logging.info(f"  Left lung size: {lung_left_size}, Right lung size: {lung_right_size}, Stomach size: {stomach_size}")
        return combined_labels
    
    # Calculate lung and stomach centroids' z-coordinates
    lung_left_centroid = ndimage.center_of_mass(lung_left_largest)
    lung_right_centroid = ndimage.center_of_mass(lung_right_largest)
    stomach_centroid = ndimage.center_of_mass(stomach_largest)
    
    lung_z = (lung_left_centroid[2] + lung_right_centroid[2]) / 2
    stomach_z = stomach_centroid[2]
    
    # Calculate z-axis distance between stomach centroid and lung centroid
    stomach_lung_z_distance = abs(stomach_z - lung_z)
    logging.info(f"Stomach centroid z: {stomach_z:.2f}, Lung centroid z: {lung_z:.2f}, Distance: {stomach_lung_z_distance:.2f}")
    
    # Analyze pancreas connected components
    labeled_pancreas, num_pancreas = ndimage.label(pancreas_mask)
    
    if num_pancreas <= 1:
        logging.info("Pancreas has only one connected component, no need for merging")
        return combined_labels
    
    # Calculate size of each pancreas connected component
    pancreas_sizes = ndimage.sum(pancreas_mask, labeled_pancreas, range(1, num_pancreas + 1))
    max_pancreas_idx = np.argmax(pancreas_sizes) + 1
    max_pancreas_size = pancreas_sizes[max_pancreas_idx - 1]
    
    # Create new label image
    new_labels = combined_labels.copy()
    
    # Clear original pancreas region
    new_labels[pancreas_mask] = 0
    
    # Add largest pancreas connected component
    max_pancreas_mask = (labeled_pancreas == max_pancreas_idx)
    new_labels[max_pancreas_mask] = ORGAN_LABEL_MAP['pancreas']
    
    # Process other connected components
    pancreas_merge_count = 0
    
    for i in range(1, num_pancreas + 1):
        if i == max_pancreas_idx:  # Skip largest connected component
            continue
        
        component = (labeled_pancreas == i)
        component_size = pancreas_sizes[i - 1]
        component_centroid = ndimage.center_of_mass(component)
        component_z = component_centroid[2]
        
        # Calculate z-axis distance between this connected component centroid and lung centroid
        component_lung_z_distance = abs(component_z - lung_z)
        
        # Determine if this connected component needs to be merged into other organs
        if component_lung_z_distance < stomach_lung_z_distance:
            # Centroid closer to lungs, may need to be merged into other organs
            # Find non-pancreas organs adjacent to this connected component
            dilated = ndimage.binary_dilation(component)
            border = dilated & ~component
            
            neighbor_labels = np.unique(combined_labels[border])
            # Exclude background (0) and all pancreas-related labels
            neighbor_labels = neighbor_labels[(neighbor_labels > 0) & 
                                            ~np.isin(neighbor_labels, pancreas_labels)]
            
            if len(neighbor_labels) > 0:
                # Calculate contact area with each neighboring label
                contact_areas = [(label, np.sum(border & (combined_labels == label))) 
                               for label in neighbor_labels]
                # Choose label with largest contact area as merge target
                target_label = max(contact_areas, key=lambda x: x[1])[0]
                
                # Get target organ name
                target_organ = next((name for name, id in ORGAN_LABEL_MAP.items() if id == target_label), f"Unknown-{target_label}")
                
                # Merge connected component into target organ
                new_labels[component] = target_label
                pancreas_merge_count += 1
                logging.info(f"Merging pancreas component {i} (size: {component_size}, centroid z: {component_z:.2f}) into {target_organ}, z-distance:{component_lung_z_distance:.2f} < stomach distance:{stomach_lung_z_distance:.2f}")
            else:
                # If no neighboring non-pancreas organs, keep as pancreas
                new_labels[component] = ORGAN_LABEL_MAP['pancreas']
                logging.info(f"Pancreas component {i} (size: {component_size}) has no neighboring non-pancreas organs, keeping as pancreas")
        else:
            # Centroid closer to stomach, keep as pancreas
            new_labels[component] = ORGAN_LABEL_MAP['pancreas']
            logging.info(f"Keeping pancreas component {i} (size: {component_size}, centroid z: {component_z:.2f}), z-distance:{component_lung_z_distance:.2f} >= stomach distance:{stomach_lung_z_distance:.2f}")
    
    logging.info(f"Pancreas processing complete: kept largest connected component (size: {max_pancreas_size}), merged {pancreas_merge_count} connected components to other organs")
    
    return new_labels

def process_case(case_dir, output_dir, class_map_name="all"):
    """Process a single case
    
    Args:
        case_dir: Path to the case directory
        output_dir: Path to output directory (can be None for in-place processing)
        class_map_name: Name of the class map to use
    
    Returns:
        Tuple of (success, message)
    """
    case_folder = os.path.basename(os.path.normpath(case_dir))
    try:
        start_time = time.time()
        success = fix_liver_segmentation(case_dir, output_dir, class_map_name)
        elapsed_time = time.time() - start_time
        
        if success:
            result = f"{case_folder} processed successfully in {elapsed_time:.2f} seconds"
            return (True, result)
        else:
            result = f"{case_folder} processing failed after {elapsed_time:.2f} seconds"
            return (False, result)
    except Exception as e:
        return (False, f"{case_folder} processing error: {str(e)}")

def process_all_cases(input_dir, output_dir=None, num_processes=None, case_list_file=None, class_map_name="all"):
    """Process all cases using dynamic task assignment
    
    Args:
        input_dir: Directory containing all case subdirectories
        output_dir: Directory where processed results will be saved. If None,
                   results are saved in subdirectories of each case
        num_processes: Number of processes to use for parallel processing
        case_list_file: Path to txt file containing case names to process. If None, processes all cases
        class_map_name: Name of the class map to use for processing
    """
    # Validate class map
    if class_map_name not in available_class_maps:
        logging.error(f"Unknown class map: {class_map_name}. Available: {list(available_class_maps.keys())}")
        return
    
    logging.info(f"Using class map: {class_map_name}")
    
    # Get all case folders
    case_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    # Filter and sort BDMAP cases by number and prefix
    bdmap_cases = []
    for folder in case_folders:
        # More flexible BDMAP format checking - support both numeric and alphanumeric formats
        if folder.startswith('BDMAP_'):
            try:
                # Extract the part after 'BDMAP_'
                id_part = folder.split('_')[1]
                
                # Check if it's pure numeric (original format)
                if id_part.isdigit():
                    case_number = int(id_part)
                    sort_key = (0, case_number, '')  # (type, number, prefix)
                    bdmap_cases.append((sort_key, folder))
                    logging.debug(f"Found numeric BDMAP case: {folder} -> {case_number}")
                
                # Check if it's alphanumeric format (A/V prefix + numbers)
                elif len(id_part) > 1 and id_part[0].isalpha() and id_part[1:].isdigit():
                    prefix = id_part[0]
                    case_number = int(id_part[1:])
                    # Sort: A cases first (type=1), then V cases (type=2), then by number
                    type_order = 1 if prefix == 'A' else 2 if prefix == 'V' else 3
                    sort_key = (type_order, case_number, prefix)
                    bdmap_cases.append((sort_key, folder))
                    logging.debug(f"Found alphanumeric BDMAP case: {folder} -> {prefix}{case_number}")
                
                else:
                    logging.warning(f"Unknown BDMAP case format: {folder}")
                    continue
                    
            except (ValueError, IndexError) as e:
                logging.warning(f"Invalid BDMAP case format: {folder} - {str(e)}")
                continue
    
    # Sort by the sort key: first by type (numeric=0, A=1, V=2), then by number, then by prefix
    bdmap_cases.sort(key=lambda x: x[0])
    
    # Convert to the format expected by the rest of the code
    bdmap_cases_formatted = []
    for sort_key, folder in bdmap_cases:
        type_order, case_number, prefix = sort_key
        if type_order == 0:  # Numeric format
            display_id = case_number
        else:  # Alphanumeric format
            display_id = f"{prefix}{case_number}"
        bdmap_cases_formatted.append((display_id, folder))
    
    bdmap_cases = bdmap_cases_formatted
    
    # Debug: Show first few cases found and case type distribution
    if bdmap_cases:
        logging.info(f"Found {len(bdmap_cases)} total BDMAP cases")
        
        # Count different types
        numeric_count = sum(1 for case_id, _ in bdmap_cases if isinstance(case_id, int))
        a_count = sum(1 for case_id, _ in bdmap_cases if isinstance(case_id, str) and case_id.startswith('A'))
        v_count = sum(1 for case_id, _ in bdmap_cases if isinstance(case_id, str) and case_id.startswith('V'))
        
        logging.info(f"Case type distribution: Numeric: {numeric_count}, A-prefix: {a_count}, V-prefix: {v_count}")
        
        sample_cases = [folder for _, folder in bdmap_cases[:5]]
        logging.info(f"Sample cases (first 5): {sample_cases}")
        
        # Show range information
        first_case_id, first_case_name = bdmap_cases[0]
        last_case_id, last_case_name = bdmap_cases[-1]
        logging.info(f"Case range: {first_case_name} (ID: {first_case_id}) to {last_case_name} (ID: {last_case_id})")
    else:
        logging.error("No BDMAP cases found at all")
        # Show all folders for debugging
        logging.info(f"All folders in directory: {case_folders[:10]}")
        return
    
    # Determine target cases based on case_list_file parameter
    if case_list_file is not None:
        # Read case names from txt file
        try:
            with open(case_list_file, 'r', encoding='utf-8') as f:
                specified_cases = [line.strip() for line in f if line.strip()]
            logging.info(f"Read {len(specified_cases)} case names from {case_list_file}")
            
            # Filter bdmap_cases to only include those specified in the txt file
            target_cases = [(case_id, name) for case_id, name in bdmap_cases if name in specified_cases]
            
            # Check for cases that were specified but not found
            found_cases = {name for _, name in target_cases}
            missing_cases = [case for case in specified_cases if case not in found_cases]
            if missing_cases:
                logging.warning(f"The following cases from txt file were not found in input directory: {missing_cases}")
            
            logging.info(f"Processing {len(target_cases)} cases specified in txt file")
        except FileNotFoundError:
            logging.error(f"Case list file not found: {case_list_file}")
            return
        except Exception as e:
            logging.error(f"Error reading case list file {case_list_file}: {str(e)}")
            return
    else:
        # Default: process all BDMAP cases found in the directory
        target_cases = bdmap_cases
        logging.info(f"Processing all {len(target_cases)} BDMAP cases found in directory")
    
    # Extract sorted case folder names
    sorted_case_folders = [folder for _, folder in target_cases]
    total_cases = len(sorted_case_folders)
    
    if total_cases == 0:
        if case_list_file is not None:
            logging.error("No valid cases found matching the specified case list file")
        else:
            logging.error("No valid BDMAP cases found in the input directory")
            # List available cases for debugging
            available_cases = [folder for _, folder in bdmap_cases[:10]]  # Show first 10
            if available_cases:
                logging.info(f"Available cases (first 10): {available_cases}")
                # Show the range of case IDs found
                case_ids = [case_id for case_id, _ in bdmap_cases]
                logging.info(f"Available case IDs (first 10): {case_ids[:10]}")
        return
    
    logging.info(f"Found {total_cases} valid cases to process")
    if sorted_case_folders:
        first_case_id = next((case_id for case_id, name in target_cases if name == sorted_case_folders[0]), "Unknown")
        last_case_id = next((case_id for case_id, name in target_cases if name == sorted_case_folders[-1]), "Unknown")
        logging.info(f"Processing range: {sorted_case_folders[0]} (ID: {first_case_id}) to {sorted_case_folders[-1]} (ID: {last_case_id})")
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output will be saved to: {output_dir}")
    
    # Create log files in appropriate location
    log_dir = output_dir if output_dir is not None else input_dir
    
    # Main summary log
    summary_log = os.path.join(log_dir, "postprocessing_summary.log")
    file_handler = logging.FileHandler(summary_log, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Create separate processing order log
    processing_order_log = os.path.join(log_dir, "processing_order.log")
    
    # Determine number of processes to use
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    num_processes = min(num_processes, multiprocessing.cpu_count(), total_cases)
    
    logging.info(f"Starting DYNAMIC PARALLEL processing of {total_cases} cases from {input_dir} using {num_processes} processes")
    logging.info("Each CPU will process one case at a time, then pick up the next available case")
    
    # Create full case paths in sorted order
    case_paths = [os.path.join(input_dir, case_folder) for case_folder in sorted_case_folders]
    
    # Write processing order information to separate log file
    import datetime
    with open(processing_order_log, 'w', encoding='utf-8') as order_file:
        order_file.write("=" * 80 + "\n")
        order_file.write("ORGAN SEGMENTATION POST-PROCESSING - PROCESSING ORDER LOG\n")
        order_file.write("=" * 80 + "\n")
        order_file.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        order_file.write(f"Processing Mode: DYNAMIC PARALLEL (one case per CPU, dynamic assignment)\n")
        order_file.write(f"Number of Processes: {num_processes}\n")
        order_file.write(f"Input Directory: {input_dir}\n")
        order_file.write(f"Output Directory: {output_dir if output_dir else 'Same as input (after_processing subdirs)'}\n")
        order_file.write(f"Class Map: {class_map_name}\n")
        order_file.write(f"Case List File: {case_list_file if case_list_file else 'None (processing all found cases)'}\n")
        order_file.write("-" * 80 + "\n")
        order_file.write(f"TOTAL CASES TO PROCESS: {total_cases}\n")
        order_file.write("-" * 80 + "\n")
        
        # Write all discovered cases with better formatting
        order_file.write(f"ALL DISCOVERED BDMAP CASES ({len(bdmap_cases)}):\n")
        for i, (case_id, case_name) in enumerate(bdmap_cases, 1):
            status = "SELECTED" if case_name in sorted_case_folders else "SKIPPED"
            order_file.write(f"  {i:4d}. {case_name:<20} (ID: {case_id:<10}) - {status}\n")
        
        order_file.write("\n" + "-" * 80 + "\n")
        order_file.write(f"CASES SCHEDULED FOR PROCESSING ({total_cases}):\n")
        order_file.write("-" * 80 + "\n")
        
        for i, case_path in enumerate(case_paths, 1):
            case_name = os.path.basename(case_path)
            case_id = next((case_id for case_id, name in target_cases if name == case_name), "Unknown")
            order_file.write(f"  {i:4d}. {case_name:<20} (ID: {case_id})\n")
        
        order_file.write("\n" + "-" * 80 + "\n")
        order_file.write("PROCESSING RESULTS (updated in real-time):\n")
        order_file.write("-" * 80 + "\n")
    
    # Record the cases to be processed (in console and main log)
    logging.info("Cases to be processed (in order):")
    for i, case_path in enumerate(case_paths, 1):
        case_name = os.path.basename(case_path)
        case_id = next((case_id for case_id, name in target_cases if name == case_name), "Unknown")
        logging.info(f"  {i:3d}. {case_name} (ID: {case_id})")
    
    logging.info(f"Processing order details saved to: {processing_order_log}")
    
    # Use partial to fix the output_dir and class_map_name parameters
    process_func = partial(process_case, output_dir=output_dir, class_map_name=class_map_name)
    
    # DYNAMIC PARALLEL PROCESSING using as_completed pattern
    try:
        logging.info("Starting dynamic parallel processing...")
        start_time = time.time()
        
        success_count = 0
        failure_count = 0
        completed_cases = {}  # Dictionary to track completed cases by case_name
        
        # Use ProcessPoolExecutor for better control
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all tasks and create a mapping from future to case info
            future_to_case = {}
            for i, case_path in enumerate(case_paths):
                case_name = os.path.basename(case_path)
                case_id = next((case_id for case_id, name in target_cases if name == case_name), "Unknown")
                
                future = executor.submit(process_func, case_path)
                future_to_case[future] = {
                    'index': i + 1,
                    'case_name': case_name,
                    'case_id': case_id,
                    'case_path': case_path
                }
            
            logging.info(f"Submitted {len(future_to_case)} tasks to {num_processes} workers")
            logging.info("Workers will pick up cases dynamically as they complete previous ones")
            
            # Process results as they complete (not in order)
            for future in as_completed(future_to_case):
                case_info = future_to_case[future]
                case_name = case_info['case_name']
                case_id = case_info['case_id']
                case_index = case_info['index']
                
                try:
                    success, message = future.result()
                    
                    completed_cases[case_name] = {
                        'index': case_index,
                        'case_name': case_name,
                        'case_id': case_id,
                        'success': success,
                        'message': message
                    }
                    
                    if success:
                        success_count += 1
                        logging.info(f"✓ [{success_count + failure_count:3d}/{total_cases}] {case_name} (ID: {case_id}) - {message}")
                    else:
                        failure_count += 1
                        logging.error(f"✗ [{success_count + failure_count:3d}/{total_cases}] {case_name} (ID: {case_id}) - {message}")
                    
                    # Update processing order log immediately
                    with open(processing_order_log, 'a', encoding='utf-8') as order_file:
                        status_symbol = "✓" if success else "✗"
                        status_text = "SUCCESS" if success else "FAILED"
                        completion_time = datetime.datetime.now().strftime('%H:%M:%S')
                        order_file.write(f"  {case_index:4d}. {status_symbol} {case_name:<20} (ID: {case_id:<10}) - {status_text} [{completion_time}]\n")
                        order_file.write(f"       {message}\n")
                    
                    # Show progress
                    completed = success_count + failure_count
                    logging.info(f"Progress: {completed}/{total_cases} completed ({completed/total_cases*100:.1f}%), Success: {success_count}, Failed: {failure_count}")
                    
                except Exception as e:
                    failure_count += 1
                    error_message = f"{case_name} processing exception: {str(e)}"
                    logging.error(f"✗ [{success_count + failure_count:3d}/{total_cases}] {error_message}")
                    
                    completed_cases[case_name] = {
                        'index': case_index,
                        'case_name': case_name,
                        'case_id': case_id,
                        'success': False,
                        'message': error_message
                    }
                    
                    # Update processing order log immediately
                    with open(processing_order_log, 'a', encoding='utf-8') as order_file:
                        completion_time = datetime.datetime.now().strftime('%H:%M:%S')
                        order_file.write(f"  {case_index:4d}. ✗ {case_name:<20} (ID: {case_id:<10}) - EXCEPTION [{completion_time}]\n")
                        order_file.write(f"       {error_message}\n")
        
        end_time = time.time()
        total_processing_time = end_time - start_time
        logging.info(f"Dynamic parallel processing completed in {total_processing_time:.2f} seconds")
        
    except KeyboardInterrupt:
        logging.error("Processing interrupted by user")
        # Remove log file handler
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        return
    
    # Sort completed cases by original index for final summary
    results_details = []
    for case_name in sorted_case_folders:
        if case_name in completed_cases:
            results_details.append(completed_cases[case_name])
        else:
            # This shouldn't happen, but just in case
            case_id = next((case_id for case_id, name in target_cases if name == case_name), "Unknown")
            results_details.append({
                'index': len(results_details) + 1,
                'case_name': case_name,
                'case_id': case_id,
                'success': False,
                'message': f"{case_name} - Not processed (unknown error)"
            })
    
    # Update processing order log with final summary
    with open(processing_order_log, 'a', encoding='utf-8') as order_file:
        order_file.write("\n" + "=" * 80 + "\n")
        order_file.write("PROCESSING SUMMARY:\n")
        order_file.write("=" * 80 + "\n")
        order_file.write(f"Processing Mode: DYNAMIC PARALLEL\n")
        order_file.write(f"Number of Workers: {num_processes}\n")
        order_file.write(f"Total Cases: {total_cases}\n")
        order_file.write(f"Successful: {success_count} ({success_count/total_cases*100:.1f}%)\n")
        order_file.write(f"Failed: {failure_count} ({failure_count/total_cases*100:.1f}%)\n")
        order_file.write(f"Total Processing Time: {total_processing_time:.2f} seconds\n")
        order_file.write(f"Average Time per Case: {total_processing_time/total_cases:.2f} seconds\n")
        order_file.write(f"Completion Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if failure_count > 0:
            order_file.write("\nFAILED CASES:\n")
            order_file.write("-" * 40 + "\n")
            for result in results_details:
                if not result['success']:
                    order_file.write(f"  {result['case_name']} (ID: {result['case_id']}): {result['message']}\n")
    
    # Final Summary
    logging.info("=" * 60)
    logging.info("DYNAMIC PARALLEL PROCESSING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Processing complete. Total: {total_cases}, Success: {success_count}, Failure: {failure_count}")
    logging.info(f"Total processing time: {total_processing_time:.2f} seconds")
    logging.info(f"Average time per case: {total_processing_time/total_cases:.2f} seconds")
    
    if success_count > 0:
        logging.info(f"Successfully processed cases: {success_count}/{total_cases} ({success_count/total_cases*100:.1f}%)")
    if failure_count > 0:
        logging.warning(f"Failed cases: {failure_count}/{total_cases} ({failure_count/total_cases*100:.1f}%)")
    
    logging.info(f"Detailed processing order and results saved to: {processing_order_log}")
    
    # Remove summary log file handler
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Organ segmentation post-processing tool")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Input directory containing case folders with segmentations")
    parser.add_argument("--output", "-o", type=str, required=False, default=None,
                        help="Output directory where processed results will be saved")
    parser.add_argument("--processes", "-p", type=int, required=False, default=None,
                        help="Number of processes to use (default: number of CPU cores)")
    parser.add_argument("--case_list", "-c", type=str, required=False, default=None,
                        help="Path to txt file containing case names to process (one per line). If not specified, processes BDMAP_00000001 to BDMAP_00001000")
    parser.add_argument("--class_map", "-m", type=str, required=False, default="all",
                        choices=list(available_class_maps.keys()),
                        help="Class map to use for processing (default: all)")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug mode to show detailed folder information")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    input_dir = args.input
    output_dir = args.output
    num_processes = args.processes
    case_list_file = args.case_list
    class_map_name = args.class_map
    
    # Set debug logging level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir if output_dir else 'Same as input (in after_processing subdirectories)'}")
    logging.info(f"Number of processes: {num_processes if num_processes else 'Auto (using all available CPU cores)'}")
    logging.info(f"Class map: {class_map_name}")
    if case_list_file:
        logging.info(f"Case list file: {case_list_file}")
    else:
        logging.info(f"Processing all BDMAP cases found in directory")
    
    process_all_cases(input_dir, output_dir, num_processes, case_list_file, class_map_name)
    logging.info("Post-processing complete")
