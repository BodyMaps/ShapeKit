from utils import *

data_type = np.int16
from config import class_map, organ_adjacency_map






def post_processing_liver(segmentation_dict):
    """
    Post-processing for liver.
    
    Main problems for liver:
        * artifacts
        * holes 
        * disconnected parts
    """
    liver_mask = segmentation_dict['liver'].copy()
    
    # keep only the main components
    cleaned_liver_mask = suppress_non_largest_components_binary(liver_mask, keep_top=3)
    cleaned_liver_mask = remove_small_components(liver_mask, np.sum(cleaned_liver_mask)/10)
    
    # Update
    segmentation_dict['liver'] = cleaned_liver_mask

    return segmentation_dict


def post_processing_pancreas(segmentation_dict, dice_threshold=0.05):
    """
    Post-processing for pancreas. 
    
    Main problems for pancreas:
        * artifacts
        * incomplete construction
    
    In some cases, there are large artifacts that will influence the surppass process
    in a heavy scale. Hence, need to compare with other organ parts (tail, body, head).
    """

    pancreas_mask = segmentation_dict['pancreas'].copy()
    try:    # is pancreas sub-parts exist, check with 

        head_mask     = segmentation_dict['pancreas_head']
        body_mask     = segmentation_dict['pancreas_body']
        tail_mask     = segmentation_dict['pancreas_tail']

        # Combine parts
        combined_parts_mask = ((head_mask + body_mask + tail_mask) > 0).astype(data_type)
        dice = soft_dice(pancreas_mask, combined_parts_mask)
        
        if dice > dice_threshold:
            # uppress non largest components
            new_pancreas_mask = suppress_non_largest_components_binary(pancreas_mask)

        else:
            # replace with combined one
            new_pancreas_mask = combined_parts_mask

    except:
        
        new_pancreas_mask = suppress_non_largest_components_binary(pancreas_mask, keep_top=2)

    
    # update
    segmentation_dict['pancreas'] = new_pancreas_mask
    
    return segmentation_dict


def post_processing_colon_intestine(segmentation_dict):
    """
    Post-processing for colons. 
    
    Main problems for pancreas:
        * small artifacts
        * redundant structure
    """
    colon_mask = segmentation_dict.get('colon')
    intestine_mask = segmentation_dict.get('intestine')

    # remove small artifacts
    cleaned_colon_mask = remove_small_components(colon_mask, threshold=np.sum(colon_mask)/10)
    cleaned_intestine_mask = remove_small_components(intestine_mask, threshold=np.sum(intestine_mask)/10)

    # re-insert
    segmentation_dict['colon'] = cleaned_colon_mask
    segmentation_dict['intestine'] = cleaned_intestine_mask
    
    return segmentation_dict



def post_processing_stomach(segmentation_dict):
    """
    Post processing for stomach.

        * in some cases, there are food inside stomach, therefore needs to fill

        
    """
    stomach_mask = segmentation_dict['stomach'].copy()

    # detect weather it is filled with food
    stomach_mask_filled = fill_holes_3d(stomach_mask)

    if np.sum(stomach_mask_filled) < 2 * np.sum(stomach_mask):
        # needs to fill
        stomach_mask = stomach_mask_filled
    
    cleaned_stomach_mask = remove_small_components(mask=stomach_mask, threshold=np.sum(stomach_mask) / 10)

    segmentation_dict['stomach'] = cleaned_stomach_mask

    return segmentation_dict


def post_processing_duodenum(segmentation_dict):
    """ New """
    duodenum_mask = segmentation_dict.get('duodenum')
    cleaned_duodenum_mask = remove_small_components(duodenum_mask, np.sum(duodenum_mask)/10)
    segmentation_dict['duodenum'] = cleaned_duodenum_mask

    return segmentation_dict

def post_processing_spleen(segmentation_dict):
    """
    Post-prcessing for spleen.

    Main problems for spleen:
        * artifacts
        * disconnected
    """
    spleen_mask = segmentation_dict['spleen'].copy()

    cleaned_spleen_mask = suppress_non_largest_components_binary(spleen_mask, 2)

    segmentation_dict['spleen'] = cleaned_spleen_mask

    return segmentation_dict


def check_z_reverse(segmentation_dict, AXIS_Z, check_organ_1='kidney_left', check_organ_2='lung_left')->bool:
    """
    Check if the Z-axis order of organs is reversed in the segmentation.

    This function compares the mean Z positions of two organs (default: left kidney and left lung)
    in a segmentation mask to determine if the order along the Z axis is anatomically reversed.
    - In normal CT orientation, the lung (or stomach) should have a lower mean Z than the kidney.
    - If the lung (or stomach) appears "below" the kidney (lung_z > kidney_z), this suggests a Z-axis flip.

    Notes:
        - If the lung mask is missing or empty, the function tries the stomach mask as a backup.
        - If the kidney mask is missing or empty, the function skips the check and returns False.
        - Issues a warning if either organ is missing or empty.
    """
    kidney_mask = segmentation_dict.get(check_organ_1)
    lung_mask = segmentation_dict.get(check_organ_2)

    if kidney_mask is None or not np.any(kidney_mask):
        print("[WARNING] Kidney mask not found or empty. Skipping z check.")
        return False

    if lung_mask is None or not np.any(lung_mask):
        print("[WARNING] Lung mask not found or empty. Use stomach instead.")
        lung_mask = segmentation_dict.get('stomach')
        

    # Detect Z-axis direction
    lung_z = np.mean(np.argwhere(lung_mask)[:, AXIS_Z])
    kidney_z = np.mean(np.argwhere(kidney_mask)[:, AXIS_Z])
    reversed_z = lung_z > kidney_z  # Normally lung_z(stomach) < kidney_z in CTs

    
    return reversed_z


def check_organ_location(segmentation_dict, organ_mask, organ_name, AXIS_Z, reference='kidney_left'):
    """
    Check some extreme oragn locations for anatomical boundaries.

        * suitable for femur, bladder and prostate. set kidney as reference organ.
    
    """

    reference_mask = segmentation_dict.get(reference)
    reversed_z = check_z_reverse(segmentation_dict, AXIS_Z)
    print(f"[INFO] Checking {organ_name}..., Z-axis reversed: {reversed_z}")

    try:
        z_limit = np.mean(np.argwhere(reference_mask)[:, AXIS_Z])
    except:
        print(f"[Info] Organ location check failed with {reference}, now try liver")
        reference_mask = segmentation_dict.get('liver')

        try:
            z_limit = np.mean(np.argwhere(reference_mask)[:, AXIS_Z])
        except:
            print(f"[Info] Still failed, skiping ...")
            return organ_mask
        

    # Prepare to filter femur voxels
    corrected_organ_mask = organ_mask.copy()
    organ_coords = np.argwhere(organ_mask)

    for coord in organ_coords:
        z = coord[AXIS_Z]
        if (not reversed_z and z < z_limit) or (reversed_z and z > z_limit):
            corrected_organ_mask[tuple(coord)] = 0

    removed_voxels = np.sum(organ_mask) - np.sum(corrected_organ_mask)
    if removed_voxels > 0:
        print(f"[INFO] Removed {removed_voxels} invalid {organ_name} voxels above reference {reference}")

    return corrected_organ_mask


def post_processing_femur(segmentation_dict: dict, axis_map: dict, calibration_standards_mask: np.ndarray) -> dict:
    """
    Post-processing for right and left femur masks stored in a segmentation_dict.
    
    """
    # Get masks
    femur_right = segmentation_dict.get("femur_right", None)
    femur_left = segmentation_dict.get("femur_left", None)

    if femur_right is None or femur_left is None:
        print("[WARNING] Femur masks not found in segmentation_dict.")
        return segmentation_dict

    # Merge and clean
    femur_mask = ((femur_left > 0) | (femur_right > 0)).astype(np.uint8)

    femur_mask =  check_organ_location(segmentation_dict, femur_mask, 'femur', axis_map['z'])

    # set as 0
    if not np.any(femur_mask):

        right_mask = np.zeros(shape=femur_left.shape)
        left_mask = np.zeros(shape=femur_right.shape)
        segmentation_dict['femur_right'] = right_mask
        segmentation_dict['femur_left'] = left_mask

        return segmentation_dict

    # first step cleaning
    cleaned_mask = remove_small_components(femur_mask, np.sum(femur_mask)/10)

    # Split right / left
    right_mask, left_mask = split_right_left(mask=cleaned_mask, AXIS=axis_map['x'])

    # Reassign if needed using liver
    right_mask, left_mask = reassign_left_right_based_on_liver(
        right_mask,
        left_mask,
        liver_mask=calibration_standards_mask
    )

    # Update dictionary
    segmentation_dict['femur_right'] = right_mask
    segmentation_dict['femur_left'] = left_mask

    return segmentation_dict


def post_processing_kidney(segmentation_dict: dict, axis_map: dict, calibration_standards_mask: np.ndarray) -> dict:
    """
    Post-prcessing for right and left kidney.

    Main problems for kidney:
        * artifacts
        * holes
        * right / left not fully seperated
    """

    kidney_left = segmentation_dict.get("kidney_left", None)
    kidney_right = segmentation_dict.get("kidney_right", None)

    if kidney_left is None or kidney_right is None:
        print("[WARNING] Kidney masks not found in segmentation_dict.")
        return segmentation_dict

    # Combine and clean
    kidney_mask = ((kidney_left > 0) | (kidney_right > 0)).astype(np.uint8)
    cleaned_kidney_mask = remove_small_components(kidney_mask, threshold=np.sum(kidney_mask) / 10)

    # Split left/right
    right_mask, left_mask = split_right_left(cleaned_kidney_mask, AXIS=axis_map['x'])

    # Reassign if necessary using liver reference
    right_mask, left_mask = reassign_left_right_based_on_liver(
        right_mask,
        left_mask,
        liver_mask=calibration_standards_mask
    )

    # Keep only largest component per side
    right_mask = suppress_non_largest_components_binary(right_mask, keep_top=1)
    left_mask = suppress_non_largest_components_binary(left_mask, keep_top=1)

    # Update dict
    segmentation_dict['kidney_left'] = left_mask
    segmentation_dict['kidney_right'] = right_mask

    return segmentation_dict



# the biggest problem is here, how to deal with the extreme-shape problem with lung?

def dongli_lung_constraints(segmentation_dict, axis_map, 
                              target_label: str = 'lung_left', fallback_label: str = 'colon', min_size: int = 50):
    """
    @ Dongli He 

    Post-processing for mislabelled lung components based on anatomical constraints.

    Returns:
        dict: Updated segmentation dictionary.
    """

    assert target_label in ['lung_left', 'lung_right'], "Target must be 'lung_left' or 'lung_right'"
    lung_mask = segmentation_dict.get(target_label, None)

    # use kidney and stomach to judege whether z is in a reversed sequence
    reversed_z = check_z_reverse(
        segmentation_dict, 
        axis_map['z'],
        check_organ_1='kidney_left',
        check_organ_2='stomach')

    if lung_mask is None or not np.any(lung_mask):
        return segmentation_dict

    # Combine reference organs for Z-bound check
    reference_mask = (
        segmentation_dict.get('kidney_left', 0) |
        segmentation_dict.get('kidney_right', 0)  |
        segmentation_dict.get('spleen', 0) |
        segmentation_dict.get('colon', 0) |
        segmentation_dict.get('intestine', 0)  
    ).astype(np.uint8)

    if not np.any(reference_mask):
        print(f"[INFO] Reference organs not found for {target_label}. Skipping.")
        return segmentation_dict


    Z = axis_map['z']
    z_lower_bound = np.mean(np.argwhere(reference_mask)[:, Z])

    cc_map = cc3d.connected_components(lung_mask, connectivity=6)
    new_lung_mask = np.zeros_like(lung_mask, dtype=np.uint8)
    fallback_mask = segmentation_dict.get(fallback_label, np.zeros_like(lung_mask, dtype=np.uint8))

    # filter out the extreme cases
    for cc_id in np.unique(cc_map):
        if cc_id == 0 or np.sum(cc_map == cc_id) < min_size:
            continue
        
        coords = np.argwhere(cc_map == cc_id)

        z_max = np.mean(coords[:, Z]) # use average position instead

        if (reversed_z and z_max < z_lower_bound) or (not reversed_z and z_max > z_lower_bound):
            fallback_mask[tuple(coords.T)] = 1
            print(f"[INFO] Reassigned {target_label} component to {fallback_label}")
        else:
            new_lung_mask[cc_map == cc_id] = 1


    segmentation_dict[target_label] = new_lung_mask
    segmentation_dict[fallback_label] = fallback_mask

    return segmentation_dict




def post_processing_lung(segmentation_dict: dict, axis_map: dict, calibration_standards_mask: np.ndarray) -> dict:
    """
    Post-prcessing for right and left lung.

    Main problems for lung:
        * right / left not fully seperated

    
    ** extreme cases:
        -- in some situation, there are some severely mislabelled lung,
                use anatomical constraints to re-assign
            
    """
    
    # for the case lung is not in the abdominal reference area
    # segmentation_dict = dongli_lung_constraints(segmentation_dict, axis_map, 'lung_left')
    # segmentation_dict = dongli_lung_constraints(segmentation_dict, axis_map, 'lung_right')

    lung_left = segmentation_dict.get("lung_left", None)
    lung_right = segmentation_dict.get("lung_right", None)

    if lung_left is None or lung_right is None:
        print("[WARNING] Missing lung_left or lung_right in segmentation_dict.")
        return segmentation_dict

    lung_mask = ((lung_left > 0) | (lung_right > 0)).astype(np.uint16)
    lung_mask = suppress_non_largest_components_binary(lung_mask, keep_top=2)


    # If lung got fully reassigned
    if np.sum(lung_mask) == 0:
        print("[INFO] No valid lung remaining after reassignment. Skipping lung post-processing.")
        segmentation_dict['lung_left'] = np.zeros_like(lung_mask, dtype=np.uint8)
        segmentation_dict['lung_right'] = np.zeros_like(lung_mask, dtype=np.uint8)
        return segmentation_dict

    # Split left and right lungs
    right_mask, left_mask = split_right_left(lung_mask, AXIS=axis_map['x'])

    # Fallback split if unbalanced
    volume_ratio = np.sum(right_mask) / (np.sum(left_mask) + 1e-5)
    if volume_ratio > 2 or volume_ratio < 0.5:
        print("[INFO] Unbalanced lung split detected. Using fallback split_organ().")
        right_mask, left_mask = split_organ(mask=lung_mask, axis=axis_map['x'])

    # Align left/right assignment based on liver position
    right_mask, left_mask = reassign_left_right_based_on_liver(
        right_mask,
        left_mask,
        liver_mask=calibration_standards_mask
    )

    # Clean small artifacts
    right_mask = remove_small_components(right_mask, threshold=np.sum(right_mask) / 10)
    left_mask = remove_small_components(left_mask, threshold=np.sum(left_mask) / 10)

    # Update dict
    segmentation_dict['lung_left'] = left_mask
    segmentation_dict['lung_right'] = right_mask

    return segmentation_dict



def post_processing_bladder_prostate(segmentation_dict:dict, segmentation:np.array, axis=2):
    """
    Post-processing for bladder and prostate.

    Keeps components that:
        - Fall within the Z range between kidneys and femurs (regardless of axis direction)
        - For prostate: must lie below the bladder in Z
    """
    target_organs = ['bladder', 'prostate']

    for organ in target_organs:
        organ_mask = segmentation_dict.get(organ).copy()
        organ_mask = remove_small_components(organ_mask, np.sum(organ_mask)/10)
        if np.sum(organ_mask) == 0:
            continue
        
        # check location
        cleaned_mask = check_organ_location(segmentation_dict, organ_mask, 
                                            organ_name=organ, AXIS_Z=axis)

        segmentation_dict[organ] = cleaned_mask
        
    return segmentation_dict


def post_processing_aorta_postcava(segmentation_dict:dict):
    """
    Post-processing for aorta and postcava.

    Main errors:
        * small artifacts
    
    """

    target_organs = ['aorta', 'postcava']
    for organ in target_organs:
        organ_mask = segmentation_dict.get(organ)
        cleaned_organ_mask = remove_small_components(organ_mask,np.sum(organ_mask)/10)
        segmentation_dict[organ] = cleaned_organ_mask

    return segmentation_dict



from scipy.spatial import KDTree
def reassign_false_positives(segmentation_dict: dict, organ_adjacency_map: dict, check_size_threshold=2000):
    organ_centers = {}
    organ_masks = {}

    # Cache organ centers
    for organ in organ_adjacency_map:
        mask = segmentation_dict.get(organ, None)
        if mask is None or np.sum(mask) == 0:
            continue
        center = compute_center(mask)
        if center is not None:
            organ_masks[organ] = mask
            organ_centers[organ] = np.array(center)

    # Build KDTree for all centers (used on a per-organ basis)
    for organ, mask in organ_masks.items():
        organ_center = organ_centers[organ]
        cc_map = cc3d.connected_components(mask, connectivity=6)
        updated_mask = np.zeros_like(mask)

        check_size_threshold = np.sum(mask)/10
        # Prepare KDTree for adjacent organs
        adj_organs = [adj for adj in organ_adjacency_map[organ] if adj in organ_centers]
        if adj_organs:
            adj_centers = np.array([organ_centers[adj] for adj in adj_organs])
            adj_tree = KDTree(adj_centers)
        else:
            adj_tree = None

        for cc_id in np.unique(cc_map):
            if cc_id == 0:
                continue
            cc_mask = (cc_map == cc_id)
            if np.sum(cc_mask) < check_size_threshold:
                continue
            
            cc_center = np.array(compute_center(cc_mask))
            if cc_center is None:
                continue

            # Compare distances using KDTree
            dist_self = np.linalg.norm(cc_center - organ_center)

            if adj_tree is None:
                updated_mask[cc_mask] = 1
                continue

            dist_adj, idx_adj = adj_tree.query(cc_center)
            if dist_adj < dist_self:
                adj_organ = adj_organs[idx_adj]
                if adj_organ not in segmentation_dict:
                    segmentation_dict[adj_organ] = np.zeros_like(mask)
                segmentation_dict[adj_organ][cc_mask] = 1
                print(f"[INFO] Reassigned component from {organ} → {adj_organ}")
            else:
                updated_mask[cc_mask] = 1

        segmentation_dict[organ] = updated_mask

    return segmentation_dict

    

def post_processing_adrenal_gland(segmentation_dict: dict, axis_map: dict, calibration_standards_mask: np.ndarray) -> dict:
    """
    Post-processing for adrenal glands: assigns correct left/right using liver position.
    """
    adrenal_left = segmentation_dict.get("adrenal_gland_left", None)
    adrenal_right = segmentation_dict.get("adrenal_gland_right", None)

    if adrenal_left is None or adrenal_right is None:
        print("[WARNING] Missing adrenal gland masks in segmentation_dict.")
        return segmentation_dict

    adrenal_mask = ((adrenal_left > 0) | (adrenal_right > 0)).astype(np.uint8)
    adrenal_mask = remove_small_components(adrenal_mask, np.sum(adrenal_mask) / 10)

    # Split left/right
    right_mask, left_mask = split_right_left(adrenal_mask, AXIS=axis_map['x'])
    
    # right and left
    right_mask, left_mask = reassign_left_right_based_on_liver(
        right_mask,
        left_mask,
        calibration_standards_mask
    )

    segmentation_dict['adrenal_gland_left'] = left_mask
    segmentation_dict['adrenal_gland_right'] = right_mask

    return segmentation_dict