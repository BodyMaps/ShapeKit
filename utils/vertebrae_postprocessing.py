import numpy as np
import nibabel as nib
import os
import cc3d
import copy
from tqdm import tqdm
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label, binary_dilation, binary_erosion
from skimage.measure import label, regionprops
from scipy import ndimage


#### @jliu452 postprocessing codes for the vertabreas part
#### TODO Not finished warning


# the general mapping
all_labels = {
    26: "vertebrae_L5",
    27: "vertebrae_L4",
    28: "vertebrae_L3",
    29: "vertebrae_L2",
    30: "vertebrae_L1",
    31: "vertebrae_T12",
    32: "vertebrae_T11",
    33: "vertebrae_T10",
    34: "vertebrae_T9",
    35: "vertebrae_T8",
    36: "vertebrae_T7",
    37: "vertebrae_T6",
    38: "vertebrae_T5",
    39: "vertebrae_T4",
    40: "vertebrae_T3",
    41: "vertebrae_T2",
    42: "vertebrae_T1",
    43: "vertebrae_C7",
    44: "vertebrae_C6",
    45: "vertebrae_C5",
    46: "vertebrae_C4",
    47: "vertebrae_C3",
    48: "vertebrae_C2",
    49: "vertebrae_C1"
}



def get_index_arr(img):
    return np.moveaxis(np.moveaxis(np.stack(np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]))),0,3),0,1)


def remove_small_components(mask, threshold):
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    for region in regions:
        if region.area < threshold:
            labeled_mask[labeled_mask == region.label] = 0
    return labeled_mask > 0


def fill_holes(mask):
    return ndimage.binary_fill_holes(mask)


def fill(segmentation):
    """
    Fill the small holes inside vertebrae structure
    """
    replace_segmentation = np.zeros_like(segmentation)
    unique_labels = np.unique(segmentation)
    for label_id in unique_labels:
        if label_id == 0:
            continue

        mask = (segmentation == label_id).astype(int)  
        mask = fill_holes(mask)
        replace_segmentation[mask] = label_id

    return replace_segmentation


def split_overmerged_triplets(merged_segmentation, size_dict, label_z_centers, counter, size_threshold_ratio=1.5):
    """
    Split over-merged vertebrae based on a triplet rule:
    If label i is much larger than min(label i-1, i-2), we split it by Z-axis.
    Skips any deleted/merged labels.
    """
    sorted_labels = sorted(size_dict.keys(), reverse=True)  # from label 24 → 1
    next_new_label = np.max(merged_segmentation) + 1

    for i in range(2, len(sorted_labels)):
        i2, i1, i0 = sorted_labels[i-2], sorted_labels[i-1], sorted_labels[i]

        if i0 not in size_dict or i1 not in size_dict or i2 not in size_dict:
            continue  # skip if any of the three labels were removed

        size0 = size_dict[i0]
        size1 = size_dict[i1]
        size2 = size_dict[i2]

        threshold = size_threshold_ratio * min(size1, size2)
        if size0 > threshold and counter > 0: # ensure not exceed limit
            
            # Candidate for splitting
            mask = merged_segmentation == i0
            coords = np.argwhere(mask)
            if coords.shape[0] == 0:
                continue
            coords_upper, coords_lower = wise_split_vertebra(coords)

            # Apply splitting
            for voxel in coords_upper:
                merged_segmentation[tuple(voxel)] = i0  # keep original label
            for voxel in coords_lower:
                merged_segmentation[tuple(voxel)] = next_new_label  # assign new label

            print(f"[INFO] (Vertebrae Module) Label {i0} was too large → split into {i0} (upper) + {next_new_label} (lower)")

            # Update label dictionaries
            size_dict[i0] = len(coords_upper)
            size_dict[next_new_label] = len(coords_lower)
            label_z_centers[i0] = np.median(coords_upper[:, 2])
            label_z_centers[next_new_label] = np.median(coords_lower[:, 2])

            next_new_label += 1
            counter -= 1

    return merged_segmentation, label_z_centers


def wise_split_vertebra(coords):
    """
    Split voxels into two equal parts along the Z-axis.
    
    TODO add ai-learnt splitor in the future
    """
    # Sort by Z descending (top to bottom)
    sorted_coords = coords[np.argsort(coords[:, 2])[::-1]]

    half = len(sorted_coords) // 2
    coords_upper = sorted_coords[:half]
    coords_lower = sorted_coords[half:]

    return coords_upper, coords_lower


def relabel_by_z_order(segmentation, label_z_centers, start_label=1):
    """
    Renames labels based on Z-axis center ordering (bottom to top).
    """

    sorted_labels = sorted(label_z_centers.items(), key=lambda x: x[1], reverse=False)

    new_segmentation = segmentation.copy()
    label_mapping = {}
    new_label_z_centers = {}

    for new_id, (old_id, z_center) in enumerate(sorted_labels, start=start_label):
        new_segmentation[segmentation == old_id] = new_id
        label_mapping[old_id] = new_id
        new_label_z_centers[new_id] = z_center  # keep the same z_center, but update label key

    print(f"[INFO]  (Vertebrae Module) Relabeled {len(label_mapping)} labels from Z-bottom to Z-top.")
    return new_segmentation, new_label_z_centers


def balance_protrusion(segmentation, label_z_centers, min_cc_voxel=1000):
    """
    examine the subsegmentations in a pair-group (A, B). A is bottom(smaller label) and B is top(bigger label). 
    if found that A's sub-region center is over B's center, merge into B.
    or if found that B's sub-region center is below A's center, merge into A.
    """
    corrected_seg = segmentation.copy()
    sorted_labels = sorted(label_z_centers.keys())

    for i in range(len(sorted_labels) - 1):
        A = sorted_labels[i]
        B = sorted_labels[i + 1]
        z_A = label_z_centers[A]
        z_B = label_z_centers[B]

        # Process components in A
        cc_map_A = cc3d.connected_components(corrected_seg == A, connectivity=6)
        for cc_id in np.unique(cc_map_A):
            if cc_id == 0:
                continue
            coords = np.argwhere(cc_map_A == cc_id)
            if coords.shape[0] < min_cc_voxel:
                continue
            z_median = np.median(coords[:, 2])
            if z_median > z_B:
                print(f"[INFO] Sub-region of label {A} protrudes into {B}, reassigning.")
                for voxel in coords:
                    corrected_seg[tuple(voxel)] = B

        # Process components in B
        cc_map_B = cc3d.connected_components(corrected_seg == B, connectivity=6)
        for cc_id in np.unique(cc_map_B):
            if cc_id == 0:
                continue
            coords = np.argwhere(cc_map_B == cc_id)
            if coords.shape[0] < min_cc_voxel:
                continue
            z_median = np.median(coords[:, 2])
            if z_median < z_A:
                print(f"[INFO] Sub-region of label {B} drops into {A}, reassigning.")
                for voxel in coords:
                    corrected_seg[tuple(voxel)] = A

    return corrected_seg


def reallocate_based_on_size(segmentation):
    """
    deal with the extra-small and extra-large subsegmentations
    """
    size_dict = {}
    label_z_centers = {}
    unique_labels = np.unique(segmentation)
    print(unique_labels)
    for label_id in unique_labels:
        if label_id == 0:
            continue
        # label_id += 26 # NOTE set the starting index
        mask = segmentation == label_id
        mask = remove_small_components(mask, threshold=np.sum(mask)/10) 

        coords = np.argwhere(mask)
        if coords.shape[0] == 0:
            continue
        z_center = np.median(coords[:, 2])
        label_z_centers[label_id] = z_center
        size_dict[label_id] = np.sum(mask)


    size_threshold_ratio = 2/3
    # Step 2: Find the unusual small one and merge
    merged_segmentation = segmentation.copy()
    need_to_merge_label = []

    for label_id, _ in label_z_centers.items():
        try:
            if size_dict[label_id] < size_threshold_ratio * (size_dict[label_id-1] + size_dict[label_id+1])/2:
                # obvious small one, need to merge with nearest neighbor
                need_to_merge_label.append(label_id)
                
        except:
            continue

    split_counter = len(need_to_merge_label)
    for label_id in need_to_merge_label:
            
            min_dist = np.inf
            nearest_label = None
            z_center = label_z_centers[label_id]
            for other_id, other_z in label_z_centers.items():
                if other_id == label_id:
                    continue
                dist = abs(z_center - other_z)
                if dist < min_dist:
                    min_dist = dist
                    nearest_label = other_id

            print(f"[INFO] (Vertebrae Module) Label {label_id} merged into {nearest_label}")
            merged_segmentation[merged_segmentation == label_id] = nearest_label

            size_dict[nearest_label] += size_dict[label_id]
            del size_dict[label_id] # remove the merged
            del label_z_centers[label_id]
        

    # Step 3: Find the unusual large one and split into 2 parts, forming new label
    print(f"[INFO] (Vertebrae Module) Total {split_counter} splits need to be made")
    # now we examine the remaining labels we use triplet. i-2, i-1. i. when i is larger than the 1.5*min(size(i-2), size(i-1)), then need split
    split_segmentation, label_z_centers = split_overmerged_triplets(
        merged_segmentation, 
        size_dict, 
        label_z_centers, 
        counter= split_counter,
        size_threshold_ratio=1.5)

    # print("New Z-centers locations:", label_z_centers)

    # Step 4: Re-schedule the label
    new_segmentation, label_z_centers = relabel_by_z_order(split_segmentation, label_z_centers)

    # Step 5: Protrusion balance
    new_segmentation = balance_protrusion(new_segmentation, label_z_centers)

    return new_segmentation



def merge_cc_of_adjacent(cc_cur, cc_above, voxel_supression_threshold):
    
    
    nof_voxels_cc = [(x, np.sum(cc_cur == x)) for x in np.unique(cc_cur)]
    relevant_cc = []

    for idx, nof_voxels in nof_voxels_cc:
        if nof_voxels > voxel_supression_threshold:
            relevant_cc.append((idx, nof_voxels))
    
    # Remove background cc from relevant cc. Assumption is that background is largest cc
    relevant_cc = sorted(relevant_cc, key=lambda x: x[1], reverse=True)[1:]
    
    nof_voxels_above = [(x, np.sum(cc_above == x )) for x in np.unique(cc_above)]

    relevant_cc_above = []
    for idx, nof_voxels in nof_voxels_above:
        if nof_voxels > voxel_supression_threshold:
            relevant_cc_above.append((idx, nof_voxels))
        # Do not supress small components here, as they will be handeled at the vertebra itself
    
    #Ignore the largest non background_cc component as it well be the vertebra itself
    relevant_cc_above = sorted(relevant_cc_above, key=lambda x: x[1], reverse=True)[2:]

    #There are components left from the vertebra which are neither background nor the vertebra itself
    if len(relevant_cc_above) > 0:
        #Pool the remaining components above with all relevant cc of current vertebra 
        mskcc_pool = np.zeros(cc_cur.shape).astype(np.bool_)
        for idx, _ in relevant_cc_above:
            mskcc_pool = np.logical_or(mskcc_pool, cc_above==idx)
        for idx, _ in relevant_cc:
            mskcc_pool = np.logical_or(mskcc_pool, cc_cur == idx)

        cc_pool = cc3d.connected_components(mskcc_pool)
        rel_components_pool = sorted([(x, np.sum(cc_pool == x )) for x in np.unique(cc_pool)],key=lambda x:x[1], reverse=True)[1:]

        return cc_pool==rel_components_pool[0][0]
    
    else:
        return None


def get_relevant_ccs(cc, keep_threshold, keep_main=True):

    if keep_main:
        cutoff_idx = 1
    else:
        cutoff_idx = 2
    return sorted([(x,np.sum(cc==x)) for x in np.unique(cc) if np.sum(cc==x) > keep_threshold],key=lambda x:x[1], reverse=True)[cutoff_idx:]


def spine_adjacent_pairs(img, voxel_supression_threshold=10, default_val=0):
    """
    Check alternating connected component to identfy fractins assigned to the wrong vertebra
    """
    labels = list(all_labels.keys())
    mod_img = copy.deepcopy(img)
    
    #Get triplets of adjacent vertebras
    triplets = []
    for l in range(len(labels)):
        # Regular triplet
        if l > 0 and l < len(labels)-1:
            triplets.append((labels[l-1], labels[l], labels[l+1]))
        # First triplet
        elif l<len(labels)-1:
            assert l == 0, "Just to be sure" #TODO: Remove before release
            triplets.append((labels[l], labels[l+1]))
        # Last triplet
        elif l>0:
            assert l==len(labels)-1, "Just to be sure" #TODO: Remove before release
            triplets.append((labels[l-1], labels[l]))
    
    for idx, triplet in enumerate(triplets):
        # print(f"[INFO] Processing triplet no. {idx}/{len(triplets)}")
        #Seperately handel first and last triplet
        if idx==0 or idx==len(triplets)-1:
            current, below = triplet
            above = None
        elif idx == len(triplets)-1:
            above, current = triplet
            below = None
        #Standard triplet
        else:
            above, current, below = triplet
            msk_cur = mod_img == current
            cc_cur = cc3d.connected_components(msk_cur)
            
            #Supress small connectred components
            nof_voxels_cc = [(x, np.sum(cc_cur == x)) for x in np.unique(cc_cur)]
            relevant_cc = []

            for idx, nof_voxels in nof_voxels_cc:
                if nof_voxels > voxel_supression_threshold:
                    relevant_cc.append((idx, nof_voxels))
                else:
                    #Set fragments smaller than voxel_supression_threshold to background
                    mod_img[cc_cur == idx] = default_val
            
            # Remove background cc from relevant cc
            background_index = sorted(relevant_cc, key=lambda x: x[1], reverse=True)[0]
            relevant_cc.remove(background_index)

            if above is not None:
                msk_above = mod_img == above
                cc_above = cc3d.connected_components(msk_above, connectivity=6)
                rel_cc_above = get_relevant_ccs(cc_above,keep_threshold=voxel_supression_threshold, keep_main=False)
            
            if below is not None:
                msk_below = mod_img == below
                cc_below = cc3d.connected_components(msk_below, connectivity=6)
                rel_cc_below = get_relevant_ccs(cc_below,keep_threshold=voxel_supression_threshold, keep_main=False)
            
            if above is not None and len(rel_cc_above) > 0:
                
                consolidated_vetebra_above = merge_cc_of_adjacent(cc_cur, cc_above, voxel_supression_threshold=voxel_supression_threshold)
                if consolidated_vetebra_above is not None:
                    mod_img[consolidated_vetebra_above] = current
                     
            
            elif below is not None and len(rel_cc_below) > 0:
                consolidated_vetebra_below = merge_cc_of_adjacent(cc_cur, cc_below, voxel_supression_threshold=voxel_supression_threshold)
                if consolidated_vetebra_below is not None:
                    mod_img[consolidated_vetebra_below] = current
    return mod_img  


def supress_non_largest_components(img, default_val = 0):
    """supress all non largest components"""
    
    index_arr = get_index_arr(img)
    img_mod = copy.deepcopy(img)
    new_background = np.zeros(img.shape, dtype=np.bool_)
    for name, _ in all_labels.items():

            # print(f"[INFO] Now processing supress non largest cc on {label}")
            label_cc = cc3d.connected_components(img == name, connectivity=6)
            uv, uc = np.unique(label_cc, return_counts=True)
            dominant_vals = uv[np.argsort(uc)[::-1][:2]]
            if len(dominant_vals)>=2: #Case: no predictions
                new_background = np.logical_or(new_background, np.logical_not(np.logical_or(label_cc==dominant_vals[0], label_cc==dominant_vals[1])))

    for voxel in index_arr[new_background]:
        img_mod[tuple(voxel)] = default_val

    return img_mod


def postprocessing_vertebrae(segmentation_dict: dict):
    """
    Post-processing for vertebrae labels.

    Steps:
        1. Reallocate label IDs based on size (e.g. largest → most important label)
        2. Suppress all non-largest connected components
        3. Fill holes within vertebrae volumes
    """

    # Step 1: Stack all vertebrae segmentations into a single 3D array
    vertebrae_labels = [key for key in segmentation_dict if key.startswith('vertebrae_')]
    print(vertebrae_labels)
    if len(vertebrae_labels) == 0:
        print("[INFO] (Vertebrae Module) Vertebraes not found, skipping ...")
        return segmentation_dict
    
    vertebrae_segmentations = np.zeros_like(next(iter(segmentation_dict.values())), dtype=np.uint8)
    for idx, vertebra_name in enumerate(sorted(vertebrae_labels), start=1):
        
        # NOTE !! the starting index
        idx += 26
        mask = segmentation_dict[vertebra_name]
        vertebrae_segmentations[mask > 0] = idx  # Assign unique label per vertebra, and initial

    # Reallocate labels based on size
    segmentation = reallocate_based_on_size(vertebrae_segmentations)
    segmentation = spine_adjacent_pairs(segmentation, voxel_supression_threshold=100)

    # Suppress non-largest components (remove noise)
    segmentation = supress_non_largest_components(np.array(segmentation))

    # Fill internal holes
    segmentation = fill(segmentation)

    # Split back into individual vertebrae masks
    processed_dict = segmentation_dict
    for label_id in np.unique(segmentation):
        if label_id == 0:
            continue
        vertebra_name = vertebrae_labels[label_id - 1]
        processed_dict[vertebra_name] = (segmentation == label_id).astype(np.uint8)

    return processed_dict