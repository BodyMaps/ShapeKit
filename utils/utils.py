import numpy as np
import nibabel as nib
import os
import cc3d
import copy
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label, binary_fill_holes, binary_dilation, binary_erosion, binary_closing, center_of_mass
from skimage.morphology import disk, convex_hull_image
from skimage.measure import label, regionprops
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from nibabel.orientations import aff2axcodes
import gc
from copy import deepcopy
import yaml
from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
)


####################################################################################
# utils.py - Organ Segmentation Post-Processing Utilities
#
# Description:
# This module provides utility functions for cleaning, analyzing, splitting, smoothing,
# and exporting 3D medical organ segmentation masks (typically in NIfTI format).
# Designed to facilitate robust downstream processing of multi-organ segmentations,
# these utilities address common issues such as label confusion, anatomical symmetry,
# component fragmentation, and mask hollowing.
#
# Core Functionalities:
# ----------------------------------------------------------------------------------
# • compute_center(mask):
#     - Computes the center of mass of a 3D binary mask.
#
# • component_location_info(labels_out, component_labels):
#     - Computes centroid coordinates and distances for each connected component.
#
# • filter_cc(organ_mask):
#     - Removes connected components far from the main centroid (artifact suppression).
#
# • fill_holes(mask), fill_holes_3d(mask):
#     - Fill 2D/3D holes within binary masks to ensure solid volumes.
#
# • fill_convex_hull_2p5D(mask):
#     - Fills each 2D slice with its convex hull, regularizing mask shapes slice-wise.
#
# • suppress_non_largest_components_binary(mask, keep_top):
#     - Keeps only the largest N connected components in a mask.
#
# • remove_small_components(mask, threshold):
#     - Removes connected components smaller than a voxel count threshold.
#
# • split_organ(mask, axis):
#     - Splits a symmetric organ mask (e.g., lung) into left/right or top/bottom.
#
# • split_right_left(mask, AXIS):
#     - Separates organ masks into left/right using coordinate medians and connectivity.
#
# • balance_protrusion_between_masks(mask_A, mask_B, axis, min_cc_voxel):
#     - Resolves overlaps and label confusion between adjacent/touching masks.
#
# • smooth_binary_image(binary_image, iterations):
#     - Morphological smoothing of binary images via dilation/erosion.
#
# • smooth_segmentation(segmentation):
#     - Smooths each label in a multi-label segmentation independently.
#
# • soft_dice(mask1, mask2, sigma):
#     - Computes a soft Dice similarity with optional Gaussian smoothing.
#
# • reinsert_organ(organ_mask, organ_index, segmentation):
#     - Reinserts an updated organ mask into a multi-organ segmentation volume.
#
# • reassign_left_right_based_on_liver(right_mask, left_mask, liver_mask):
#     - Assigns right/left mask identities based on anatomical liver proximity.
#
# • get_axis_map(img):
#     - Maps physical axes (e.g., L/R, A/P, S/I) to array axes from a NIfTI image.
#
# • bbox_distance(mask1, mask2):
#     - Calculates minimum bounding box separation between two masks.
#
# • read_all_segmentations(folder_path, data_type):
#     - Loads all 3D segmentation masks from a folder into a dictionary.
#
# • save_and_combine_segmentations(processed_segmentation_dict, class_map, ...):
#     - Saves individual organ segmentations and combines them into a single label map.
#
####################################################################################






def compute_center(mask):
    # use in-built center_of_mass to boost execution speed
    if np.sum(mask) == 0:
        return None
    return np.array(center_of_mass(mask))


def component_location_info(labels_out, component_labels)->np.array:

    """
    Get the location info for every connected component's distance from centroid (0, 0, 0).
    
    """
    distances = []
    centroids = []
    for label in component_labels:
        
        coords = np.argwhere(labels_out == label)
        centroid = coords.mean(axis=0)
        centroids.append(centroid)
        distance = np.linalg.norm(centroid)  # distance to (0,0,0)
        distances.append(distance)

    distances = np.array(distances)
    
    return distances


def filter_cc(organ_mask):
    """
    Filter out the cc that locats far away from centroid
    
    """
    labels_out = cc3d.connected_components(organ_mask)
    
    # 1st clean artifacts
    component_labels = np.unique(labels_out)
    component_labels = component_labels[component_labels != 0]
    distances = component_location_info(labels_out, component_labels)
    average_distance = distances.mean()

    if len(distances)  > 2:
        # keep only within the average distance
        keep_mask = np.zeros_like(labels_out, dtype=bool)
        for label, distance in zip(component_labels, distances):
            if distance <= average_distance:
                keep_mask |= (labels_out == label)
        new_organ_mask = keep_mask
        
    else:
        
        new_organ_mask = organ_mask

    return new_organ_mask


def fill_holes(mask):
    """
    Fill in small 2d holes.
    """

    return ndimage.binary_fill_holes(mask)



def fill_holes_3d(mask):
    structure = generate_binary_structure(3, 2)
    closed = binary_closing(mask, structure=structure)
    filled = binary_fill_holes(closed)
    return filled.astype(mask.dtype)


def fill_convex_hull_2p5D(mask_3d):
    """
    Applies convex hull filling slice-by-slice along the Z-axis (height).
    
    """
    filled_mask = np.zeros_like(mask_3d)

    for z in range(mask_3d.shape[0]):  # iterate over Z (height axis)
        slice_2d = mask_3d[z]
        if np.any(slice_2d):  # only process non-empty slices
            filled_mask[z] = convex_hull_image(slice_2d).astype(mask_3d.dtype)

    return filled_mask



def suppress_non_largest_components_binary(mask, keep_top=2):
    """
    Suppress all but the top-N largest connected components in a binary mask.
    Args:
        mask (np.ndarray): Binary mask where 1 indicates foreground.
        keep_top (int): Number of largest components to keep. Default is 2.

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    if not np.any(mask):
        return mask  

    # Label connected components
    label_cc = cc3d.connected_components(mask.astype(np.uint8), connectivity=6)
    labels_all, counts_all = np.unique(label_cc, return_counts=True)
    nonzero_mask = labels_all != 0
    labels = labels_all[nonzero_mask]
    counts = counts_all[nonzero_mask]
    
    # Get top-N largest components
    if len(labels) > keep_top:
        top_labels = labels[np.argsort(counts)[::-1][:keep_top]]
    else:
        top_labels = labels

    cleaned_mask = np.isin(label_cc, top_labels).astype(mask.dtype)
    return cleaned_mask


def smooth_binary_image(binary_image, iterations=1):
    """
    Smoothing tools
    """
    num_dimensions = binary_image.ndim
    structure = generate_binary_structure(num_dimensions, 1)  # Structure for the number of dimensions
    smoothed_image = binary_dilation(binary_image, structure=structure, iterations=iterations)
    smoothed_image = binary_erosion(smoothed_image, structure=structure, iterations=iterations)

    return smoothed_image


def smooth_segmentation(segmentation):
    """
    General smoothing.
    """
    smoothed_segmentation = np.zeros_like(segmentation)

    unique_labels = np.unique(segmentation)
    for label_id in tqdm(unique_labels, desc='[INFO] smoothing'):
        if label_id == 0:
            continue

        mask = (segmentation == label_id).astype(int) 
        smoothed_mask = smooth_binary_image(mask)
        smoothed_segmentation[smoothed_mask] = label_id  

    return smoothed_segmentation


def soft_dice(mask1, mask2, sigma=1.0):
    """Dice after Gaussian smoothing to reduce edge sensitivity"""
    mask1_blur = gaussian_filter(mask1.astype(np.float32), sigma)
    mask2_blur = gaussian_filter(mask2.astype(np.float32), sigma)
    intersection = (mask1_blur * mask2_blur).sum()
    return 2. * intersection / (mask1_blur.sum() + mask2_blur.sum() + 1e-8)


def reinsert_organ(organ_mask, organ_index, segmentation):
    """
    Reinsert into segmentation
    """
    
    segmentation[segmentation == organ_index] = 0
    segmentation[organ_mask == 1] = organ_index

    return segmentation


def remove_small_components(mask, threshold):
    """
    Removes small connected components from a binary mask.

    Args:
        mask (np.array): Binary 3D array.
        threshold (int): Minimum number of voxels a component must have to be kept.

    Returns:
        np.array: A cleaned binary mask.
    """
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    for region in regions:
        if region.area < threshold:
            labeled_mask[labeled_mask == region.label] = 0
    return labeled_mask > 0


def split_right_left(mask, AXIS=0):
    """
    Splits a symmetric organ mask into right and left components along a specified axis.

    Each connected component in the mask is assigned to either the left or right side based 
    on the mean coordinate of its voxels along the specified axis.

    Args:
        mask (np.ndarray): Binary 3D mask containing the merged organ(s).
        AXIS (int): Axis along which to perform the split. Default is 0 (left-right).

    Returns:
        right_mask (np.ndarray): Binary mask containing right-side components.
        left_mask (np.ndarray): Binary mask containing left-side components.
    """


    coords = np.argwhere(mask == 1)
    x_mid = np.median(coords[:, AXIS])  
    left_mask  = np.zeros_like(mask, dtype=np.uint8)
    right_mask = np.zeros_like(mask, dtype=np.uint8)

    labeled_mask = cc3d.connected_components(mask, connectivity=6)
    for label in np.unique(labeled_mask):
        
        if label == 0:
            continue  # skip background
        label_coords = np.argwhere(labeled_mask == label)
        comp_center_x = label_coords[:, AXIS].mean()  
        if comp_center_x < x_mid:
            left_mask[labeled_mask == label] = 1
        else:
            right_mask[labeled_mask == label] = 1  

    return right_mask, left_mask


def split_organ(mask, axis):
    """
    Split a binary 3D organ mask into two parts by cutting at the sparsest slice
    within the central 1/3 region along a given axis.

    This function is suitable for the organs with high-symetric characters. 
    e.g. lung.

    """


    if not (0 <= axis <= 2):
        raise ValueError("Axis must be 0 (Z), 1 (Y), or 2 (X).")

    coords = np.argwhere(mask)

    # Get min and max along axis to find the full bounding box range
    axis_vals = coords[:, axis]
    axis_min = axis_vals.min()
    axis_max = axis_vals.max()
    axis_len = axis_max - axis_min + 1
    third = axis_len // 3
    central_start = axis_min + third
    central_end = axis_max - third

    slice_voxel_counts = {}
    for idx in range(central_start, central_end + 1):
        if axis == 0:
            count = np.sum(mask[idx, :, :])
        elif axis == 1:
            count = np.sum(mask[:, idx, :])
        elif axis == 2:
            count = np.sum(mask[:, :, idx])
        slice_voxel_counts[idx] = count

    # Find the slice with the fewest voxels
    cut_index = min(slice_voxel_counts, key=slice_voxel_counts.get)
    
    left_mask = np.zeros_like(mask, dtype=mask.dtype)
    right_mask = np.zeros_like(mask, dtype=mask.dtype)

    for z, y, x in coords:
        if [z, y, x][axis] < cut_index:
            left_mask[z, y, x] = 1
        else:
            right_mask[z, y, x] = 1

    return right_mask, left_mask



def plot_3d_mask(mask_3d, organ_name='lung'):
    """
    Fast 3D scatter plot of a binary lung mask using matplotlib.

    Parameters:
        mask_3d: np.ndarray, 3D binary mask (1 = organ, 0 = background)
        organ_name: str, name to show on title and filename
        sub_folder: str, subject or case ID
        save_path: str, path to save the output image
    """
    coords = np.column_stack(np.where(mask_3d > 0))  # shape (N, 3)

    

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0],  # (X, Y, Z) axis order
               c='red', marker='o', s=1, alpha=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{organ_name}')

    plt.show()



def balance_protrusion_between_masks(mask_A, mask_B, axis=2, min_cc_voxel=1000):
    """
    Adjusts overlap between two binary masks (mask_A, mask_B) based on axis-wise protrusion.
    - If a region in A extends beyond the center of B along the given axis → assign to B.
    - If a region in B drops below the center of A along the given axis → assign to A.

    Parameters:
        mask_A, mask_B : np.ndarray
            Binary masks of the two structures (same shape)
        axis : int
            Axis along which to balance (0 = Z, 1 = Y, 2 = X)
        min_cc_voxel : int
            Minimum size of a connected component to consider

    Returns:
        new_mask_A, new_mask_B : np.ndarray
            Updated binary masks (mutually exclusive)
    """
    assert mask_A.shape == mask_B.shape, "Masks must be the same shape"

    # Copy masks to avoid modifying original
    new_mask_A = mask_A.copy()
    new_mask_B = mask_B.copy()

    # Compute median centers
    z_A = np.median(np.argwhere(mask_A)[:, axis])
    z_B = np.median(np.argwhere(mask_B)[:, axis])

    # Process A: remove components that protrude into B's center
    cc_A = cc3d.connected_components(mask_A.astype(np.uint8), connectivity=6)
    for cc_id in np.unique(cc_A):
        if cc_id == 0:
            continue
        coords = np.argwhere(cc_A == cc_id)
        if coords.shape[0] < min_cc_voxel:
            continue
        z_median = np.median(coords[:, axis])
        if z_median > z_B:
            print(f"[INFO] A component protrudes into B along axis {axis}, moving to B")
            new_mask_A[cc_A == cc_id] = 0
            new_mask_B[cc_A == cc_id] = 1

    # Process B: remove components that drop into A's center
    cc_B = cc3d.connected_components(mask_B.astype(np.uint8), connectivity=6)
    for cc_id in np.unique(cc_B):
        if cc_id == 0:
            continue
        coords = np.argwhere(cc_B == cc_id)
        if coords.shape[0] < min_cc_voxel:
            continue
        z_median = np.median(coords[:, axis])
        if z_median < z_A:
            print(f"[INFO] B component drops into A along axis {axis}, moving to A")
            new_mask_B[cc_B == cc_id] = 0
            new_mask_A[cc_B == cc_id] = 1

    return new_mask_A, new_mask_B


def smooth_binary_image(binary_image, iterations=1):
    """
    Smoothing tools
    """
    num_dimensions = binary_image.ndim
    structure = generate_binary_structure(num_dimensions, 1)  # Structure for the number of dimensions
    smoothed_image = binary_dilation(binary_image, structure=structure, iterations=iterations)
    smoothed_image = binary_erosion(smoothed_image, structure=structure, iterations=iterations)

    return smoothed_image


def reassign_left_right_based_on_liver(right_mask, left_mask, liver_mask):
    """
    Reassign left and right masks based on proximity to the liver.
    Liver is assumed to always be on the right side anatomically.

    Args:
        right_mask (np.ndarray): Binary mask for the presumed right-side organ.
        left_mask (np.ndarray): Binary mask for the presumed left-side organ.
        liver_mask (np.ndarray): Binary mask for the liver (used as spatial reference).

    Returns:
        corrected_right_mask, corrected_left_mask : np.ndarray
    """
    # Compute centers of masks
    liver_center = compute_center(liver_mask)
    left_center = compute_center(left_mask)
    right_center = compute_center(right_mask)

    if liver_center is None or left_center is None or right_center is None:
        return left_mask, right_mask

    # Compute Euclidean distances between liver and left/right masks
    dist_to_left = np.linalg.norm(liver_center - left_center)
    dist_to_right = np.linalg.norm(liver_center - right_center)
    
    if dist_to_left < dist_to_right:# if left is closer to the liver, swap
        return left_mask.copy(), right_mask.copy()
    else:
        return right_mask.copy(), left_mask.copy()
    

def get_axis_map(img):
    """ 
    
    Build dict mapping physical axes to voxel axes
    """
    
    affine = img.affine
    orientation = aff2axcodes(affine)
    phys_to_std = {
        'L': 'x', 'R': 'x',
        'P': 'y', 'A': 'y',
        'I': 'z', 'S': 'z'
    }
    axis_map = {}
    for voxel_axis, code in enumerate(orientation):
        std_label = phys_to_std.get(code.upper())
        if std_label:
            axis_map[std_label] = voxel_axis
    # print(axis_map)
    
    return axis_map


def bbox_distance(mask1, mask2):
    coords1 = np.argwhere(mask1)
    coords2 = np.argwhere(mask2)
    if coords1.size == 0 or coords2.size == 0:
        return float('inf')

    min1, max1 = coords1.min(axis=0), coords1.max(axis=0)
    min2, max2 = coords2.min(axis=0), coords2.max(axis=0)

    # Compute per-axis separation
    lower_diff = min2 - max1
    upper_diff = min1 - max2
    per_axis_dist = np.maximum(lower_diff, upper_diff)
    per_axis_dist = np.maximum(per_axis_dist, 0)

    return np.linalg.norm(per_axis_dist)


def organ_HU_value(mask):
    """3d-organ HU-values"""
    return np.mean(mask)
    

def read_all_segmentations(folder_path, organ_list, subfolder_name='segmentations',
                           data_type=np.uint8, target_axcodes=('R', 'A', 'S')) -> dict:
    """
    Safely read segmentation masks from .nii.gz files, correct orientation,
    and handle corrupt or missing files gracefully.
    """
    seg_folder = os.path.join(folder_path, subfolder_name)
    segmentation_dict = {}

    if not os.path.exists(seg_folder):
        raise FileNotFoundError(f"[ERROR] Folder not found: {seg_folder}")

    files = [f for f in os.listdir(seg_folder) if f.endswith('.nii.gz')]
    if not files:
        raise ValueError(f"[ERROR] No .nii.gz files found in: {seg_folder}")

    # Find first good file for orientation reference
    ref_img = None
    for f in files:
        try:
            ref_img = nib.load(os.path.join(seg_folder, f))
            break
        except Exception as e:
            print(f"[WARNING] Failed to load {f} as reference: {e}")
            continue

    if ref_img is None:
        raise RuntimeError("[ERROR] No readable .nii.gz files found to determine orientation.")

    orig_ornt = io_orientation(ref_img.affine)
    target_ornt = axcodes2ornt(target_axcodes)
    transform = ornt_transform(orig_ornt, target_ornt)

    # Now load all valid segmentations
    for file in files:
        organ = os.path.splitext(os.path.splitext(file)[0])[0]
        if organ not in organ_list:
            continue

        file_path = os.path.join(seg_folder, file)
        try:
            nii_img = nib.load(file_path)
            arr = np.asanyarray(nii_img.dataobj).astype(data_type)

            # Apply orientation transformation
            arr = apply_orientation(arr, transform)
            if arr.ndim != 3:
                continue

            segmentation_dict[organ] = arr

        except Exception as e:
            print(f"[WARNING] Skipping {organ} due to read/format error: {e}")
            continue

        del nii_img, arr
        gc.collect() # free up memory usage

    if not segmentation_dict:
        raise RuntimeError(f"[ERROR] No valid segmentations found in: {seg_folder}")

    return segmentation_dict


def save_and_combine_segmentations(processed_segmentation_dict: dict,
                                   class_map: dict,
                                   reference_img: nib.Nifti1Image,
                                   output_folder: str,
                                   if_save_combined: False,
                                   combined_filename: str = "combined_labels.nii.gz"):
    """
    Saves all organ masks in a folder and combines them into a single label volume.

    Output_folder
      --    combined_labels
      --    segmentations
                |
                |-- livers
                |-- ...
    """
    os.makedirs(output_folder, exist_ok=True)
    seg_folder = os.path.join(output_folder, "segmentations")
    os.makedirs(seg_folder, exist_ok=True)

    # Save each organ mask individually and discard from memory
    for idx, organ in sorted(class_map.items()):
        mask = processed_segmentation_dict.pop(organ, None)
        if mask is None or not mask.any():
            continue

        mask = mask.astype(np.uint8, copy=False)
        nib.save(
            nib.Nifti1Image(mask, reference_img.affine, reference_img.header),
            os.path.join(seg_folder, f"{organ}.nii.gz")
        )
        del mask  # free memory

    # choose to save the combine label
    if if_save_combined:
        # Allocate combined volume
        sample_path = os.path.join(seg_folder, f"liver.nii.gz")
        shape = nib.load(sample_path).shape
        combined = np.zeros(shape, dtype=np.uint8)

        for idx, organ in sorted(class_map.items()):
            try:
                organ_path = os.path.join(seg_folder, f"{organ}.nii.gz")
                if not os.path.exists(organ_path):
                    continue
                mask = nib.load(organ_path).get_fdata().astype(bool)
                combined[mask] = idx
                del mask  # free memory
            except:# if some masks are null and therefore deleted, skip
                continue

        # Save combined label map
        nib.save(
            nib.Nifti1Image(combined, reference_img.affine, reference_img.header),
            os.path.join(output_folder, combined_filename)
        )