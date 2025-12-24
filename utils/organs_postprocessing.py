from utils.utils import *
import logging

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class_map = config['class_map']
class_map = {int(k): v for k, v in config['class_map'].items()}
organ_adjacency_map = config['organ_adjacency_map']
data_type = np.int16







def post_processing_liver(segmentation_dict):
    """
    Post-processing for liver.
    
    Main problems for liver:
        * artifacts
        * holes 
        * disconnected parts
    """
    liver_mask = segmentation_dict.get('liver')
    
    # keep only the main components
    cleaned_liver_mask = suppress_non_largest_components_binary(liver_mask, keep_top=3)
    
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

    pancreas_mask = segmentation_dict.get('pancreas')
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
        
        new_pancreas_mask = suppress_non_largest_components_binary(pancreas_mask)

    
    # update
    segmentation_dict['pancreas'] = new_pancreas_mask
    
    return segmentation_dict


def post_processing_colon_intestine(segmentation_dict, patient_id:str, logger:logging.Logger):
    """
    Post-processing for colons. 
    
    Main problems for pancreas:
        * small artifacts
        * redundant structure
    """
    colon_mask = segmentation_dict.get('colon')
    intestine_mask = segmentation_dict.get('intestine')

    # Compute threshold once for colon
    if colon_mask is not None and np.any(colon_mask):
        colon_threshold = max(1, np.sum(colon_mask) / 10)
        cleaned_colon_mask = remove_small_components(colon_mask, threshold=colon_threshold)
        segmentation_dict['colon'] = cleaned_colon_mask
    
    try:
        # Compute threshold once for intestine
        if intestine_mask is not None and np.any(intestine_mask):
            intestine_threshold = max(1, np.sum(intestine_mask) / 10)
            cleaned_intestine_mask = remove_small_components(intestine_mask, threshold=intestine_threshold)
            segmentation_dict['intestine'] = cleaned_intestine_mask
    except:
        logger.info(f"[INFO] {patient_id}, (Colon-Intestine Check Module) Intestine does not exist, skipped ...")
    return segmentation_dict



def post_processing_stomach(segmentation_dict):
    """
    Post processing for stomach.

        * in some cases, there are food inside stomach, therefore needs to fill

        
    """
    stomach_mask = segmentation_dict['stomach'].copy()
    if stomach_mask is not None and np.any(stomach_mask):
        stomach_threshold = max(1, np.sum(stomach_mask) / 10)
        cleaned_stomach_mask = remove_small_components(stomach_mask, threshold=stomach_threshold)
        segmentation_dict['stomach'] = cleaned_stomach_mask

    return segmentation_dict


def post_processing_duodenum(segmentation_dict):
    """"""
    duodenum_mask = segmentation_dict.get('duodenum')
    if duodenum_mask is not None and np.any(duodenum_mask):
        duodenum_threshold = max(1, np.sum(duodenum_mask) / 10)
        cleaned_duodenum_mask = remove_small_components(duodenum_mask, duodenum_threshold)
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


# def check_z_reverse(segmentation_dict, AXIS_Z, check_organ_1='kidney_left', check_organ_2='lung_left')->bool:
#     """
#     Check if the Z-axis order of organs is reversed in the segmentation.

#     This function compares the mean Z positions of two organs (default: left kidney and left lung)
#     in a segmentation mask to determine if the order along the Z axis is anatomically reversed.
#     - In normal CT orientation, the lung (or stomach) should have a lower mean Z than the kidney.
#     - If the lung (or stomach) appears "below" the kidney (lung_z > kidney_z), this suggests a Z-axis flip.

#     Notes:
#         - If the lung mask is missing or empty, the function tries the stomach mask as a backup.
#         - If the kidney mask is missing or empty, the function skips the check and returns False.
#         - Issues a warning if either organ is missing or empty.
#     """
#     kidney_mask = segmentation_dict.get(check_organ_1)
#     lung_mask = segmentation_dict.get(check_organ_2)

#     if kidney_mask is None or not np.any(kidney_mask):
#         print("[WARNING] Kidney mask not found or empty. Skipping z check.")
#         return False

#     if lung_mask is None or not np.any(lung_mask):
#         print("[WARNING] Lung mask not found or empty. Use stomach instead.")
#         lung_mask = segmentation_dict.get('stomach')
        

#     # Detect Z-axis direction
#     lung_z = np.mean(np.argwhere(lung_mask)[:, AXIS_Z])
#     kidney_z = np.mean(np.argwhere(kidney_mask)[:, AXIS_Z])
#     reversed_z = lung_z > kidney_z  # Normally lung_z(stomach) < kidney_z in CTs

    
#     return reversed_z


def check_organ_location(segmentation_dict, organ_mask, organ_name, AXIS_Z, reference='kidney_left', patient_id=None, logger=None):
    """
    Check some extreme oragn locations for anatomical boundaries.

        * suitable for femur, bladder and prostate. set kidney as reference organ.
    
    """

    reference_mask = segmentation_dict.get(reference)
    reversed_z = True
    logger.info(f"[INFO] {patient_id}, (Organ Location Check Module) Checking {organ_name}...")

    try:
        z_limit = np.mean(np.argwhere(reference_mask)[:, AXIS_Z])
    except:
        logger.info(f"[INFO] {patient_id}, (Organ Location Check Module) Organ location check failed with {reference}, now try liver")
        reference_mask = segmentation_dict.get('liver')

        try:
            z_limit = np.mean(np.argwhere(reference_mask)[:, AXIS_Z])
        except:
            logger.info(f"[INFO] {patient_id}, Still failed, skiping ...")
            return organ_mask
        

    # Vectorized filtering - much faster than looping through coordinates
    corrected_organ_mask = organ_mask.copy()
    organ_coords = np.argwhere(organ_mask)
    
    # Create boolean mask for voxels to remove
    z_values = organ_coords[:, AXIS_Z]
    if not reversed_z:
        voxels_to_remove = z_values < z_limit
    else:
        voxels_to_remove = z_values > z_limit
    
    # Apply removal using vectorized indexing
    coords_to_remove = organ_coords[voxels_to_remove]
    corrected_organ_mask[coords_to_remove[:, 0], coords_to_remove[:, 1], coords_to_remove[:, 2]] = 0

    removed_voxels = np.sum(organ_mask) - np.sum(corrected_organ_mask)
    if removed_voxels > 0:
        logger.info(f"[INFO] {patient_id}, (Organ Location Check Module) Removed {removed_voxels} invalid {organ_name} voxels above reference {reference}")

    return corrected_organ_mask


def post_processing_femur(segmentation_dict: dict, axis_map: dict, calibration_standards_mask: np.ndarray, patient_id:str, logger:logging.Logger) -> dict:
    """
    Post-processing for right and left femur masks stored in a segmentation_dict.
    
    """
    # Get masks
    femur_right = segmentation_dict.get("femur_right", None)
    femur_left = segmentation_dict.get("femur_left", None)

    if femur_right is None or femur_left is None:
        logger.info(f"[WARNING] {patient_id}, Femur masks not found in segmentation_dict.")
        return segmentation_dict

    # Merge and clean
    femur_mask = ((femur_left > 0) | (femur_right > 0)).astype(np.uint8)
    femur_mask =  check_organ_location(segmentation_dict, femur_mask, 'femur', axis_map['z'], patient_id=patient_id, logger=logger)

    # set as 0
    if not np.any(femur_mask):

        right_mask = np.zeros(shape=femur_left.shape)
        left_mask = np.zeros(shape=femur_right.shape)
        segmentation_dict['femur_right'] = right_mask
        segmentation_dict['femur_left'] = left_mask

        return segmentation_dict

    # first step cleaning
    femur_threshold = max(1, np.sum(femur_mask) / 10)
    cleaned_mask = remove_small_components(femur_mask, femur_threshold)

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


def post_processing_kidney(segmentation_dict: dict, axis_map: dict, calibration_standards_mask: np.ndarray, patient_id:str, logger:logging.Logger) -> dict:
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
        logger.info(f"[WARNING] {patient_id}, Kidney masks not found in segmentation_dict.")
        return segmentation_dict

    # Combine and clean
    kidney_mask = ((kidney_left > 0) | (kidney_right > 0)).astype(np.uint8)
    kidney_threshold = max(1, np.sum(kidney_mask) / 10)
    cleaned_kidney_mask = remove_small_components(kidney_mask, threshold=kidney_threshold)

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

def dongli_lung_constraints(
    segmentation_dict, axis_map, 
    target_label: str = 'lung_left', 
    fallback_label: str = 'colon', 
    min_size: int = 50, 
    overlap_check_threshold: float=0.8,
    patient_id: str = 'None',
    logger: logging.Logger = None,
    ):
    """
    @ Dongli He 

    Post-processing for mislabelled lung components based on anatomical constraints.

    Returns:
        dict: Updated segmentation dictionary.
    """

    assert target_label in ['lung_left', 'lung_right'], "Target must be 'lung_left' or 'lung_right'"
    lung_mask = segmentation_dict.get(target_label, None)

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

        if z_max < z_lower_bound:
            fallback_mask[tuple(coords.T)] = 1
            logger.info(f"[INFO] {patient_id}, (Lung Check Module) Reassigned {target_label} component to {fallback_label}")
        else:
            # double check with overlap ratio
            temp_mask = cc_map == cc_id
            overlap_check = np.sum(temp_mask & reference_mask) / np.sum(temp_mask)
            
            if overlap_check > overlap_check_threshold:
                fallback_mask[tuple(coords.T)] = 1
                logger.info(f"[INFO] {patient_id}, (Lung Check Module) Reassigned {target_label} component to {fallback_label}")
            else:
                # permitted
                new_lung_mask[cc_map == cc_id] = 1


    segmentation_dict[target_label] = new_lung_mask
    segmentation_dict[fallback_label] = fallback_mask

    return segmentation_dict




def post_processing_lung(segmentation_dict: dict, axis_map: dict, calibration_standards_mask: np.ndarray, patient_id: str, logger: logging.Logger) -> dict:
    """
    Post-prcessing for right and left lung.

    Main problems for lung:
        * right / left not fully seperated

    
    ** extreme cases:
        -- in some situation, there are some severely mislabelled lung,
                use anatomical constraints to re-assign
            
    """
    
    # for the case lung is not in the abdominal reference area
    segmentation_dict_new = deepcopy(segmentation_dict)
    segmentation_dict_new = dongli_lung_constraints(segmentation_dict_new, axis_map, 'lung_left', patient_id=patient_id, logger=logger)
    segmentation_dict_new = dongli_lung_constraints(segmentation_dict_new, axis_map, 'lung_right', patient_id=patient_id, logger=logger)
    
    # double-check if wrongly reassigned by location of colon and liver
    if np.sum(segmentation_dict_new.get('lung_left')) == 0 and np.sum(segmentation_dict_new.get('lung_right')) == 0 :
        colon_mask = segmentation_dict_new.get('colon')
        liver_mask = segmentation_dict_new.get('liver')

        Z = axis_map['z']
        colon_z = np.where(colon_mask)[Z]
        liver_z = np.where(liver_mask)[Z]

        z_check_bound = np.mean(colon_z)
        z_liver = np.mean(liver_z)
        if z_liver < z_check_bound:
            logger.info(f"[INFO] {patient_id}, (Lung Check Module) Ineffective lung reassignment found, fall back ...")
            del segmentation_dict_new
            gc.collect()
        else:
            logger.info(f"[INFO] {patient_id}, (Lung Check Module) No valid lung remaining after reassignment. Skip lung post-processing ...")
            return segmentation_dict_new
    else:
        # continues
        segmentation_dict = segmentation_dict_new

    lung_left = segmentation_dict.get("lung_left", None)
    lung_right = segmentation_dict.get("lung_right", None)

    if lung_left is None or lung_right is None:
        logger.info(f"[WARNING] {patient_id}, (Lung Check Module) Missing lung_left or lung_right in segmentation_dict.")
        return segmentation_dict

    lung_mask = ((lung_left > 0) | (lung_right > 0)).astype(np.uint16)
    
    lung_mask = suppress_non_largest_components_binary(lung_mask, keep_top=2)

    # Split left and right lungs
    right_mask, left_mask = split_right_left(lung_mask, AXIS=axis_map['x'])

    # Fallback split if unbalanced
    volume_ratio = np.sum(right_mask) / (np.sum(left_mask) + 1e-5)
    if volume_ratio > 2 or volume_ratio < 0.5:
        logger.info(f"[INFO] {patient_id}, (Lung Check Module) Unbalanced lung split detected. Using fallback split_organ().")
        try:
            right_mask, left_mask = split_organ(mask=lung_mask, axis=axis_map['x'])
        except:
            logger.info(f"[INFO] {patient_id}, (Lung Check Module) Fallback split_organ() failed, skip lung post-processing ...")


    # Align left/right assignment based on liver position
    right_mask, left_mask = reassign_left_right_based_on_liver(
        right_mask,
        left_mask,
        liver_mask=calibration_standards_mask
    )

    # Update dict
    segmentation_dict['lung_left'] = left_mask
    segmentation_dict['lung_right'] = right_mask

    return segmentation_dict



def post_processing_bladder_prostate(segmentation_dict:dict, segmentation:np.array, patient_id: str, logger: logging.Logger, axis=2):
    """
    Post-processing for bladder and prostate.

    Keeps components that:
        - Fall within the Z range between kidneys and femurs (regardless of axis direction)
        - For prostate: must lie below the bladder in Z
    """
    target_organs = ['bladder', 'prostate']

    for organ in target_organs:
        organ_mask = segmentation_dict.get(organ)
        if organ_mask is None:
            continue
        organ_mask = organ_mask.copy()
        organ_threshold = max(1, np.sum(organ_mask) / 10)
        organ_mask = remove_small_components(organ_mask, organ_threshold)
        if np.sum(organ_mask) == 0:
            continue
        
        # check location
        cleaned_mask = check_organ_location(
            segmentation_dict, 
            organ_mask, 
            organ_name=organ,
            AXIS_Z=axis,
            patient_id=patient_id, logger=logger)

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
        if organ_mask is None or not np.any(organ_mask):
            continue
        organ_threshold = max(1, np.sum(organ_mask) / 10)
        cleaned_organ_mask = remove_small_components(organ_mask, organ_threshold)
        segmentation_dict[organ] = cleaned_organ_mask

    return segmentation_dict


def post_processing_adrenal_gland(segmentation_dict: dict, axis_map: dict, calibration_standards_mask: np.ndarray) -> dict:
    """
    Post-processing for adrenal glands: assigns correct left/right using liver position.
    """
    adrenal_left = segmentation_dict.get("adrenal_gland_left", None)
    adrenal_right = segmentation_dict.get("adrenal_gland_right", None)

    if adrenal_left is None or adrenal_right is None:
        # print("[WARNING] Missing adrenal gland masks in segmentation_dict.")
        return segmentation_dict

    adrenal_mask = ((adrenal_left > 0) | (adrenal_right > 0)).astype(np.uint8)
    adrenal_threshold = max(1, np.sum(adrenal_mask) / 10)
    adrenal_mask = remove_small_components(adrenal_mask, adrenal_threshold)

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


def reassign_false_positives(segmentation_dict: dict, organ_adjacency_map: dict, patient_id:str, logger:logging.Logger, check_size_threshold=2000):
    """
    Reassign false positives between anatomically adjacent organs.


    Args:
        segmentation_dict (dict): A dictionary mapping organ 
        names to binary masks. 
        organ_adjacency_map (dict): A dictionary defining 
        spatial adjacency between organs. 
        check_size_threshold (int, optional): Minimum component 
        size to consider for reassignment. 
                                              
    Returns:
        dict: Updated segmentation dictionary.
    """
    organ_centers = {}
    organ_masks = {}

    # Step 1: Cache organ centers
    for organ, mask in segmentation_dict.items():
        if organ not in organ_adjacency_map:
            continue
        if mask is None or mask.sum() == 0:
            continue

        center = compute_center(mask)
        if center is not None:
            organ_masks[organ] = mask
            organ_centers[organ] = np.array(center)

    # Step 2: Iterate over each organ
    for organ, mask in organ_masks.items():
        organ_center = organ_centers[organ]
        cc_map = cc3d.connected_components(mask, connectivity=6)
        updated_mask = np.zeros_like(mask, dtype=bool)

        voxel_count = np.count_nonzero(mask)
        size_threshold = voxel_count / 10 if voxel_count > 0 else check_size_threshold

        # Prepare KDTree for adjacent organs
        adj_organs = [adj for adj in organ_adjacency_map[organ] if adj in organ_centers]
        if adj_organs:
            adj_centers = np.array([organ_centers[adj] for adj in adj_organs])
            adj_tree = KDTree(adj_centers)
        else:
            adj_tree = None

        # Get all unique component labels except background
        labels = np.unique(cc_map)
        labels = labels[labels != 0]

        for cc_id in labels:
            cc_mask = (cc_map == cc_id)

            if cc_mask.sum() < size_threshold:
                continue

            cc_center = compute_center(cc_mask)
            if cc_center is None:
                continue
            cc_center = np.array(cc_center)

            dist_self_sq = np.sum((cc_center - organ_center) ** 2)

            if adj_tree is None:
                updated_mask[cc_mask] = True
                continue

            dist_adj_sq, idx_adj = adj_tree.query(cc_center)
            dist_adj_sq = dist_adj_sq**2

            if dist_adj_sq < dist_self_sq:
                adj_organ = adj_organs[idx_adj]
                if adj_organ not in segmentation_dict:
                    segmentation_dict[adj_organ] = np.zeros_like(mask, dtype=bool)
                segmentation_dict[adj_organ][cc_mask] = True
                logger.info(f"[INFO] {patient_id}, (FP Check Module) Reassigned component from {organ} → {adj_organ}")
            else:
                updated_mask[cc_mask] = True
            
            del cc_mask  # Free temporary mask
        segmentation_dict[organ] = updated_mask.astype(mask.dtype)

        del cc_map, updated_mask  # Free memory
        gc.collect()


    return segmentation_dict