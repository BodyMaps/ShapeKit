import json
import logging
import multiprocessing
import shutil
from functools import partial
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

import cc3d
import nibabel as nib
import numpy as np
from jsonargparse import auto_cli
from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
)
from scipy.ndimage import (
    binary_erosion,
    binary_closing,
    generate_binary_structure,
)
from scipy.spatial import KDTree

from class_maps import class_maps_supported
from utils import Settings, MaskCropper, init_logger

logger = logging.getLogger(__name__)
init_logger()

intermediate_axcodes = ("R", "A", "S")

keep_fragmented_labels = {
    "aorta": 3,
    "celiac_artery": 5,
    "colon": 6,
    "common_bile_duct": 7,
    "duodenum": 8,
    "postcava": 22,
    "stomach": 25,
    "superior_mesenteric_artery": 26,
    "veins": 27,
    "esophagus": 29,
    "intestine": 31,
}

relevant_left_right_pairs = {
    "adrenal_gland": (1, 2),
    "femur": (9, 10),
    "aorta_postcava": (3, 22),
    "kidney": (12, 13),
    "lung": (15, 16),
}


class OrganRectifier:
    def __init__(self):
        super().__init__()
        self.rectified_organs = []

        # lss is short for liver, spleen, stomach
        self.lss_organs = [14, 24, 25]

        # organs that are below the lss organs
        # used in reassigning components
        self.organs_below_lss = [6, 8, 12, 13, 17, 31]

        # self.organs_no_overlap_with_stomach = [6, 8, 31]

    # entry points
    def remove_out_of_bounds_components(
        self, seg_masks: Dict[int, np.ndarray], target_label: int
    ) -> Dict[int, np.ndarray]:
        """
        Removes connected components of a target organ label that fall outside
        predefined spatial bounds (x, y, z) based on reference organs.

        Args:
            seg_masks (dict[int, np.ndarray]): Dictionary mapping class labels to binary masks.
            target_label (int): Label of the organ to process.

        Returns:
            dict[int, np.ndarray]: Updated segmentation masks with invalid components removed.
        """
        # Validate and label connected components for the target organ
        validation_result = self._validate_and_label_components(seg_masks, target_label)
        if validation_result is None or validation_result[1] < 1:
            return seg_masks  # No valid components found or no components to process

        labeled_comps, num_comps = validation_result

        # spatial limitation
        ref_bounds = self._get_reference_bounds(seg_masks, target_label)
        x_lower_bound, x_upper_bound = ref_bounds["x_lb"], ref_bounds["x_ub"]
        y_lower_bound, y_upper_bound = ref_bounds["y_lb"], ref_bounds["y_ub"]
        z_lower_bound, z_upper_bound = ref_bounds["z_lb"], ref_bounds["z_ub"]

        # Remove adrenal gland components that are outside the reference organ bounds
        removed_count = 0
        for i in range(1, num_comps + 1):
            comp_mask = labeled_comps == i
            comp_coords = np.argwhere(comp_mask)
            centroid = comp_coords.mean(axis=0)
            x, y, z = centroid
            if not (
                x_lower_bound <= x <= x_upper_bound
                and y_lower_bound <= y <= y_upper_bound
                and z_lower_bound <= z <= z_upper_bound
            ):
                seg_masks[target_label][comp_mask] = 0
                removed_count += 1

        if removed_count > 0:
            logger.debug(
                f"{removed_count} out of {num_comps} {class_maps_supported[target_label]} components outside reference organ bounds are removed."
            )
            labeled_comps, num_comps = binary_connected_components(
                seg_masks[target_label]
            )

        seg_masks = self._apply_label_specific_rules(
            seg_masks, target_label, labeled_comps, num_comps
        )

        self.rectified_organs.append(target_label)

        return seg_masks

    def reassign_components(
        self,
        seg_masks: Dict[int, np.ndarray],
        target_label: int,
        reassignment_candidates: List[int],
        min_size: int = 50,
        reassign_max_distance: int = 3,
        reassign_min_num: int = 10,
        extra_criteria: Optional[Dict] = None,
    ) -> Dict[int, np.ndarray]:
        """
        This function retains the largest component of the target label, and for other component:
        - If the component is smaller than `min_size`, it is removed.
        - Otherwise, it is reassigned to another organ label if there are enough nearby pixels from
        that organ (based on spatial distance and count thresholds).
        - Remaining unchanged components are left as is.

        Args:
            seg_masks (dict[int, np.ndarray]): A dictionary mapping class labels to binary segmentation masks.
            target_label (int): The label of the organ to process and clean up.
            reassignment_candidates (list[int]): Labels of organ classes eligible to receive reassigned components.
            min_size (int, optional): Minimum number of pixels a component must have to avoid removal. Defaults to 50.
            reassign_max_distance (int, optional): Maximum distance to consider another organ's pixels for reassignment. Defaults to 3.
            reassign_min_num (int, optional): Minimum number of close pixels from another class needed to trigger reassignment. Defaults to 10.
            extra_criteria (dict, optional): (For future use) Additional criteria for component reassignment. Defaults to None.

        Returns:
            dict[int, np.ndarray]: The updated segmentation dictionary with refined component assignments.
        """
        # Input validation
        assert min_size >= 0, "min_size should be non-negative"
        assert (
            reassign_max_distance >= 0
        ), "reassign_max_distance should be non-negative"
        assert reassign_min_num >= 0, "reassign_min_num should be non-negative"

        # Label connected components of the target organ
        validation_result = self._validate_and_label_components(seg_masks, target_label)
        if validation_result is None or validation_result[1] <= 1:
            # No valid components found or only one component
            return seg_masks

        labeled_comps, num_comps = validation_result

        # Identify the largest component to retain it as the main one
        if target_label in [12, 13]:
            # Special case for kidneys: use the largest component under adrenal glands
            largest_comp_idx = self._get_largest_comp_idx_for_kidney(
                seg_masks, target_label, labeled_comps, num_comps
            )
        else:
            comp_sizes = np.bincount(labeled_comps.ravel())[1:]
            largest_comp_idx = np.argmax(comp_sizes) + 1

        # Collect coordinates of the largest component and all reassignment candidates
        label_to_coords = {target_label: np.argwhere(labeled_comps == largest_comp_idx)}
        for candidate_label in reassignment_candidates:
            if candidate_label not in seg_masks or candidate_label == target_label:
                continue
            coords = np.argwhere(seg_masks[candidate_label])
            if coords.size > 0:
                label_to_coords[candidate_label] = coords

        reassigned_labels = []
        removed_count = 0

        if target_label in self.organs_below_lss:
            liver_coords = np.argwhere(seg_masks.get(14))
            spleen_coords = np.argwhere(seg_masks.get(24))
            stomach_coords = np.argwhere(seg_masks.get(25))

            liver_top = liver_coords[:, 2].max() if liver_coords.size != 0 else np.inf
            spleen_top = (
                spleen_coords[:, 2].max() if spleen_coords.size != 0 else np.inf
            )
            stomach_top = (
                stomach_coords[:, 2].max() if stomach_coords.size != 0 else np.inf
            )
            z_upper_bound = min(liver_top, spleen_top, stomach_top)

        # Process each component except the largest one
        for i in range(1, num_comps + 1):
            if i == largest_comp_idx:
                continue  # Skip the main component

            comp_mask = labeled_comps == i
            comp_coords = np.argwhere(comp_mask)

            # Remove the component if it's too small
            if comp_coords.size < min_size:
                seg_masks[target_label][comp_mask] = 0
                removed_count += 1
                continue

            # Check if the component can be reassigned to another organ
            reassigned = False
            tree = KDTree(comp_coords)
            for label, coords in label_to_coords.items():
                distances_to_coords = tree.query(coords)[0]  # Nearest distances
                if (
                    np.sum(distances_to_coords < reassign_max_distance)
                    >= reassign_min_num
                ):
                    reassigned = True
                    reassigned_labels.append(label)

                    # Reassign the component to the new organ label
                    if label != target_label:
                        seg_masks[target_label][comp_mask] = 0
                        seg_masks[label][comp_mask] = 1

                    break

            if target_label in self.organs_below_lss and not reassigned:
                # remove components that are above the liver/spleen/stomach
                comp_mean_z = comp_coords[:, 2].mean()
                if comp_mean_z > z_upper_bound:
                    seg_masks[target_label][comp_mask] = 0
                    removed_count += 1

        # Logging removed components
        if removed_count > 0:
            logger.debug(
                f"{removed_count} out of {num_comps} {class_maps_supported[target_label]} components outside reference organ bounds are removed."
            )

        # Logging reassigned components
        if reassigned_labels:
            logger.debug(
                f"{len(reassigned_labels)} out of {num_comps} {class_maps_supported[target_label]} components are reassigned to other organs."
            )

        return seg_masks

    ### special organs
    def apply_bladder_or_prostate_postprocessing(
        self,
        seg_masks: Dict[int, np.ndarray],
        target_label: int,
        x_threshold_ratio: float = 0.2,
    ) -> Dict[int, np.ndarray]:
        """
        Refines segmentation of bladder or prostate by removing spatially invalid components
        based on reference anatomical structures.

        Args:
            seg_masks (dict[int, np.ndarray]): Dictionary of binary masks for each organ label.
            target_label (int): Label to process (should be 4 for bladder or 23 for prostate).
            x_threshold_ratio (float): Threshold for x-coordinate deviation relative to reference width.

        Returns:
            dict[int, np.ndarray]: Updated segmentation masks.
        """
        assert target_label in [
            4,
            23,
        ], f"target_label should be 4 or 23, but got {target_label}"

        validation_result = self._validate_and_label_components(seg_masks, target_label)
        if validation_result is None or validation_result[1] < 1:
            return seg_masks

        labeled_mask, num_comps = validation_result

        # Get reference coordinates
        aorta_postcava_coords = np.argwhere(
            seg_masks.get(3, False) | seg_masks.get(22, False)
        )  # aorta/postcava
        femur_coords = np.argwhere(
            seg_masks.get(9, False) | seg_masks.get(10, False)
        )  # femur

        z_upper_bound = (
            aorta_postcava_coords[:, 2].min()
            if aorta_postcava_coords.size > 0
            else np.inf
        )

        if femur_coords.size > 0:
            x_lower_bound, x_upper_bound = (
                femur_coords[:, 0].min(),
                femur_coords[:, 0].max(),
            )
            z_lower_bound = femur_coords[:, 2].min()
        else:
            x_lower_bound, x_upper_bound = -np.inf, np.inf
            z_lower_bound = -np.inf

        # Compute mean x of aorta as reference midline
        aorta_postcava_mean_x = (
            aorta_postcava_coords[:, 0].mean() if aorta_postcava_coords.size > 0 else -1
        )

        # Estimate width from surrounding abdominal organs
        ref_coords = np.argwhere(
            seg_masks.get(14, False)
            | seg_masks.get(24, False)
            | seg_masks.get(25, False)
            | seg_masks.get(6, False)
            | seg_masks.get(15, False)
            | seg_masks.get(16, False)
        )
        ref_width = np.ptp(ref_coords[:, 0]) if ref_coords.size > 0 else np.inf

        # Filter invalid components
        removed_count = 0
        for i in range(1, num_comps + 1):
            comp_mask = labeled_mask == i
            comp_coords = np.argwhere(comp_mask)
            x, _, z = comp_coords.mean(axis=0)

            outside_bounds = not (
                x_lower_bound <= x <= x_upper_bound
                and z_lower_bound <= z <= z_upper_bound
            )
            off_center = abs(x - aorta_postcava_mean_x) > x_threshold_ratio * ref_width

            if outside_bounds or (off_center and aorta_postcava_mean_x > 0):
                seg_masks[target_label][comp_mask] = 0
                removed_count += 1

        if removed_count > 0:
            logger.debug(
                f"{removed_count} out of {num_comps} "
                f"{class_maps_supported[target_label]} components outside reference bounds are removed."
            )

        return seg_masks

    def apply_lung_postprocessing(
        self, seg_masks: Dict[int, np.ndarray], target_label: int, min_size: int = 50
    ) -> Dict[int, np.ndarray]:
        """
        Refines lung segmentation by removing or reassigning components based on spatial constraints.

        Args:
            seg_masks (dict[int, np.ndarray]): Dictionary mapping class labels to binary masks.
            target_label (int): Target label (15 for left lung, 16 for right lung).

        Returns:
            dict[int, np.ndarray]: Updated segmentation masks with invalid lung components removed or reassigned.
        """
        assert target_label in [
            15,
            16,
        ], "Target label must be either 15 (left lung) or 16 (right lung)."

        # Validate and label connected components for the target organ
        validation_result = self._validate_and_label_components(seg_masks, target_label)
        if validation_result is None or validation_result[1] < 1:
            return seg_masks  # Only one component or label is invalid; skip processing

        labeled_comps, num_comps = validation_result

        # Compute lower Z bound using liver, spleen, stomach as reference
        reference_coords = np.argwhere(
            seg_masks.get(14, False)  # liver
            | seg_masks.get(24, False)  # spleen
            | seg_masks.get(25, False)  # stomach
        )
        z_lower_bound = (
            reference_coords[:, 2].mean() if reference_coords.size > 0 else np.inf
        )

        # Identify primary lung candidate: largest component above z_lower_bound
        primary_lung_idx = -1
        max_size = 0
        for i in range(1, num_comps + 1):
            comp_mask = labeled_comps == i
            comp_coords = np.argwhere(comp_mask)
            comp_z_mean = comp_coords[:, 2].mean()
            comp_size = comp_coords.shape[0]
            if comp_z_mean >= z_lower_bound and comp_size > max_size:
                primary_lung_idx = i
                max_size = comp_size

        if primary_lung_idx == -1:
            logger.debug(
                f"No valid {class_maps_supported[target_label]} component found above abdominal organs."
            )

        reassigned_count = 0
        removed_count = 0
        fallback_label = 6  # Colon

        for i in range(1, num_comps + 1):
            if i == primary_lung_idx:
                continue

            comp_mask = labeled_comps == i
            comp_coords = np.argwhere(comp_mask)

            # Remove the component if it's too small
            if comp_coords.size < min_size:
                seg_masks[target_label][comp_mask] = 0
                removed_count += 1
                continue

            comp_z_max = comp_coords[:, 2].max()

            # Components entirely below the abdominal reference organs
            if comp_z_max < z_lower_bound:
                seg_masks[target_label][comp_mask] = 0
                seg_masks[fallback_label][comp_mask] = 1
                reassigned_count += 1

        if reassigned_count > 0:
            logger.debug(
                f"{reassigned_count} of {num_comps} {class_maps_supported[target_label]} components reassigned to {class_maps_supported[fallback_label]}."
            )

        if removed_count > 0:
            logger.debug(
                f"{removed_count} out of {num_comps} {class_maps_supported[target_label]} components outside reference organ bounds are removed."
            )

        return seg_masks

    def apply_pancreatic_duct_post_processing(
        self, seg_masks: Dict[int, np.ndarray], target_label: int
    ) -> Dict[int, np.ndarray]:
        """
        Post-process the pancreatic duct (21) by removing connected components
        that fall outside the bounding box of pancreas (17).

        Args:
            seg_masks (dict[int, np.ndarray]): Dictionary of binary segmentation masks.
            label (int): Target label for the pancreatic duct (should be 21).

        Returns:
            dict[int, np.ndarray]: Updated segmentation masks after post-processing.
        """
        assert (
            target_label == 21
        ), f"Expected label 21 for pancreatic duct, but got {target_label}"

        validation_result = self._validate_and_label_components(seg_masks, target_label)

        if validation_result is None or validation_result[1] < 1:
            # No valid components found or only one component
            return seg_masks

        labeled_comps, num_comps = validation_result

        # Define pancreas spatial bounds as reference (label 17)
        pancreas_coords = np.argwhere(seg_masks.get(17))
        if pancreas_coords.size != 0:
            x_min, x_max = pancreas_coords[:, 0].min(), pancreas_coords[:, 0].max()
            y_min, y_max = pancreas_coords[:, 1].min(), pancreas_coords[:, 1].max()
            z_min, z_max = pancreas_coords[:, 2].min(), pancreas_coords[:, 2].max()
        else:
            logger.debug(
                f"No pancreas duct found for {class_maps_supported[target_label]} post-processing."
            )
            return seg_masks

        removed_count = 0
        for comp_idx in range(1, num_comps + 1):
            comp_mask = labeled_comps == comp_idx
            comp_coords = np.argwhere(comp_mask)
            x, y, z = comp_coords.mean(axis=0)

            # Remove components outside the pancreas bounds
            if not (
                x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max
            ):
                seg_masks[target_label][comp_mask] = 0
                removed_count += 1

        if removed_count > 0:
            logger.debug(
                f"{removed_count} out of {num_comps} {class_maps_supported[target_label]} components outside pancreas bounds are removed."
            )

        return seg_masks

    ### helper functions
    def _get_reference_bounds(
        self, seg_masks: Dict[int, np.ndarray], target_label: int
    ):
        ref_bounds = {
            "x_lb": -np.inf,
            "x_ub": np.inf,
            "y_lb": -np.inf,
            "y_ub": np.inf,
            "z_lb": -np.inf,
            "z_ub": np.inf,
        }
        if target_label in [1, 2]:  # adrenal glands
            ref_coords = np.argwhere(
                (seg_masks.get(12, False))  # left kidney
                | (seg_masks.get(13, False))  # right kidney
                | (seg_masks.get(14, False))  # liver
                | (seg_masks.get(24, False))  # spleen
                | (seg_masks.get(25, False))  # stomach
            )

            if ref_coords.size != 0:
                ref_bounds["x_lb"] = ref_coords[:, 0].min()
                ref_bounds["x_ub"] = ref_coords[:, 0].max()
                ref_bounds["y_lb"] = ref_coords[:, 1].min()
                ref_bounds["y_ub"] = ref_coords[:, 1].max()
                ref_bounds["z_lb"] = ref_coords[:, 2].min()
                ref_bounds["z_ub"] = ref_coords[:, 2].max()
        elif target_label in [3, 22]:  # aorta, postcava
            ref_coords = np.argwhere(
                seg_masks.get(6, False) | seg_masks.get(31, False)
            )  # colon, intestine
            lss_coords = np.argwhere(
                (seg_masks.get(14, False))  # liver
                | (seg_masks.get(24, False))  # spleen
                | (seg_masks.get(25, False))  # stomach
            )

            if lss_coords.size != 0:
                ref_bounds["x_lb"] = lss_coords[:, 0].min()
                ref_bounds["x_ub"] = lss_coords[:, 0].max()
            if ref_coords.size != 0:
                ref_bounds["z_lb"] = ref_coords[:, 2].min()
        elif target_label in [4, 23]:  # bladder, prostate
            # bladder and prostate are dealt in apply_bladder_or_prostate_postprocessing
            pass
        elif target_label in [5, 7, 8, 12, 13, 17, 18, 19, 20, 26, 27]:  # abdomen
            # celiac artery, common bile duct, duodenum, kidneys, pancreas, superior mesenteric artery, veins
            kidney_coords = np.argwhere(
                seg_masks.get(12, False) | seg_masks.get(13, False)
            )
            liver_coords = np.argwhere(seg_masks.get(14))
            spleen_coords = np.argwhere(seg_masks.get(24))
            stomach_coords = np.argwhere(seg_masks.get(25))

            liver_top = liver_coords[:, 2].max() if liver_coords.size != 0 else np.inf
            spleen_top = (
                spleen_coords[:, 2].max() if spleen_coords.size != 0 else np.inf
            )
            stomach_top = (
                stomach_coords[:, 2].max() if stomach_coords.size != 0 else np.inf
            )

            if liver_coords.size != 0:
                ref_bounds["x_ub"] = liver_coords[:, 0].max()
            if kidney_coords.size != 0:
                ref_bounds["z_lb"] = kidney_coords[:, 2].min()
            ref_bounds["z_ub"] = min(liver_top, spleen_top, stomach_top)
        elif target_label in [6, 31]:  # colon
            # dealt in reassign_components
            ...
        elif target_label in [9, 10]:  # femur
            # can't use bladder as reference as its top is sometimes below the femurs
            ref_coords = np.argwhere(
                seg_masks.get(3, False)
                | seg_masks.get(22, False)
                | seg_masks.get(12, False)
                | seg_masks.get(13, False)
            )  # aorta, postcava, kidneys

            if ref_coords.size != 0:
                ref_bounds["z_ub"] = ref_coords[:, 2].min()
        elif target_label in [11]:  # gall bladder
            liver_coords = np.argwhere(seg_masks.get(14))  # liver
            if liver_coords.size != 0:
                ref_bounds["x_lb"] = liver_coords[:, 0].min()
                ref_bounds["x_ub"] = liver_coords[:, 0].max()
                ref_bounds["y_lb"] = liver_coords[:, 1].min()
                ref_bounds["y_ub"] = liver_coords[:, 1].max()
                ref_bounds["z_lb"] = liver_coords[:, 2].min()
                ref_bounds["z_ub"] = liver_coords[:, 2].max()
        elif target_label in [14, 24, 25]:  # liver, spleen, stomach
            kidney_coords = np.argwhere(
                seg_masks.get(12, False) | seg_masks.get(13, False)
            )
            aorta_postcava_coords = np.argwhere(
                (seg_masks.get(3, False)) | (seg_masks.get(22, False))
            )
            kidney_floor = (
                kidney_coords[:, 2].min() if kidney_coords.size != 0 else -np.inf
            )
            aorta_postcava_floor = (
                aorta_postcava_coords[:, 2].min()
                if aorta_postcava_coords.size != 0
                else -np.inf
            )

            ref_bounds["z_lb"] = max(kidney_floor, aorta_postcava_floor)
        elif target_label in [15, 16]:  # lungs
            # lungs are dealt in apply_lung_postprocessing
            pass
        elif target_label in [21]:  # pancreatic duct
            # pancreatic duct is dealt in apply_pancreatic_duct_post_processing
            pass
        elif target_label in [24, 25]:  # spleen, stomach
            kidney_coords = np.argwhere(
                seg_masks.get(12, False) | seg_masks.get(13, False)
            )
            if kidney_coords.size != 0:
                ref_bounds["z_lb"] = kidney_coords[:, 2].min()
        elif target_label in [29]:  # esophagus
            liver_coords = np.argwhere(seg_masks.get(14))
            spleen_coords = np.argwhere(seg_masks.get(24))
            stomach_coords = np.argwhere(seg_masks.get(25))

            liver_top = liver_coords[:, 2].max() if liver_coords.size != 0 else np.inf
            spleen_top = (
                spleen_coords[:, 2].max() if spleen_coords.size != 0 else np.inf
            )
            stomach_top = (
                stomach_coords[:, 2].max() if stomach_coords.size != 0 else np.inf
            )

            ref_bounds["z_lb"] = min(liver_top, spleen_top, stomach_top)
        # elif target_label in range(34, 58):
        #     ...
        else:
            logger.debug(
                f"No pre-defined reference bounds for {class_maps_supported[target_label]}"
            )

        return ref_bounds

    def _apply_label_specific_rules(
        self,
        seg_masks: Dict[int, np.ndarray],
        target_label: int,
        labeled_comps: np.ndarray,
        num_comps: int,
    ) -> Dict[int, np.ndarray]:
        if target_label in [3, 22]:  # aorta, postcava
            if 12 in seg_masks and 13 in seg_masks:  # kidney
                left_kidney_coords = np.argwhere(seg_masks[12])
                right_kidney_coords = np.argwhere(seg_masks[13])
                if left_kidney_coords.size != 0:
                    left_kidney_height = np.ptp(left_kidney_coords[:, 2])
                else:
                    left_kidney_height = -np.inf

                if right_kidney_coords.size != 0:
                    right_kidney_height = np.ptp(right_kidney_coords[:, 2])
                else:
                    right_kidney_height = -np.inf
                max_kidney_height = max(left_kidney_height, right_kidney_height)

                for i in range(1, num_comps + 1):
                    comp_mask = labeled_comps == i
                    comp_coords = np.argwhere(comp_mask)
                    comp_height = np.ptp(comp_coords[:, 2])
                    if comp_height >= max_kidney_height:
                        keep_fragmented_labels.pop(
                            class_maps_supported[target_label], None
                        )
                        break
        return seg_masks

    def _validate_and_label_components(
        self, seg_masks: Dict[int, np.ndarray], target_label: int
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Validates the presence and pixel count of a specified label in the segmentation data.
        If the label exists and has more than one connected component, label the components.

        Args:
            seg_masks (dict[int, np.ndarray]): Dictionary mapping class labels to binary masks.
            target_label (int): The label to validate and segment into connected components.

        Returns:
            - seg_masks (dict[int, np.ndarray]): If the label is missing or has <= 1 component.
            - (labeled_comps, component_count): Tuple containing labeled mask and number of components if multiple exist.
        """
        # Check if the label exists and has at least one pixel
        if target_label not in seg_masks:
            logger.debug(f"Label {target_label} not found. Skip processing.")
            return

        # Label connected components in the binary mask
        labeled_comps, num_comps = binary_connected_components(seg_masks[target_label])

        # If there's only one component
        if num_comps == 0:
            logger.debug(
                f"Label {class_maps_supported[target_label]} has no component."
            )
        elif num_comps == 1:
            logger.debug(
                f"Label {class_maps_supported[target_label]} has only one component. Skipping."
            )

        return labeled_comps, num_comps

    def _get_largest_comp_idx_for_kidney(
        self,
        seg_masks: Dict[int, np.ndarray],
        target_label: int,
        labeled_comps: np.ndarray,
        num_comps: int,
    ) -> int:
        """
        Determine the index of the largest kidney component that lies below the adrenal glands.
        If no such component exists, fall back to the globally largest component.

        Args:
            seg_masks (dict[int, np.ndarray]): Dictionary mapping label IDs to binary segmentation masks.
            target_label (int): Label ID of the kidney class being processed.
            labeled_comps (np.ndarray): Labeled mask of connected components for the kidney.
            num_comps (int): Total number of connected components found.

        Returns:
            int: Index of the largest valid component.
        """
        largest_comp_idx = -1
        largest_comp_size = 0

        if 1 in seg_masks and 2 in seg_masks:
            # Use the mean Z of the adrenal glands as an upper bound
            adrenal_gland_coords = np.argwhere(seg_masks[1] | seg_masks[2])
            z_upper_bound = (
                adrenal_gland_coords[:, 2].mean()
                if adrenal_gland_coords.size != 0
                else np.inf
            )

            for i in range(1, num_comps + 1):
                comp_mask = labeled_comps == i
                comp_coords = np.argwhere(comp_mask)
                comp_mean_z = comp_coords[:, 2].mean()
                comp_size = comp_coords.shape[0]
                if comp_mean_z <= z_upper_bound and comp_size > largest_comp_size:
                    largest_comp_idx = i
                    largest_comp_size = comp_size

        if largest_comp_idx == -1:
            if 1 not in seg_masks and 2 not in seg_masks:
                logger.debug("No adrenal glands found.")
            else:
                logger.debug(
                    f"No {class_maps_supported[target_label]} candidate found below the adrenal gland."
                )
            logger.debug(
                f"Will use the global largest component as {class_maps_supported[target_label]}."
            )
            comp_sizes = np.bincount(labeled_comps.ravel())[1:]
            largest_comp_idx = np.argmax(comp_sizes) + 1

        return largest_comp_idx


def binary_connected_components(binary_mask: np.ndarray, connectivity: int = 6):
    """
    Computes connected components in a binary mask.

    Args:
        binary_mask (np.ndarray): A binary numpy array where foreground pixels are True or 1,
                                  and background pixels are False or 0.
        connectivity (int): The connectivity to use for connected components.
                            only 4,8 (2D) and 26, 18, and 6 (3D) are allowed. Default is 6.
    Returns:
        np.ndarray: Labeled array where each connected component is assigned a unique label.
        int: The number of connected components found.
    """
    comp = cc3d.connected_components(
        binary_mask, binary_image=True, connectivity=connectivity
    )
    return comp, comp.max().item()


def extract_surface_coordinates(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extracts the coordinates of the surface (boundary) pixels of a binary mask.

    Args:
        binary_mask (np.ndarray): A binary numpy array where foreground pixels are True or 1,
                                  and background pixels are False or 0.

    Returns:
        np.ndarray: An array of coordinate tuples (row, column, ...) corresponding to the surface pixels.
    """
    mask_eroded = binary_erosion(binary_mask)
    surface_mask = binary_mask & ~mask_eroded
    return np.argwhere(surface_mask)


def calculate_median(
    seg_data, labels: Union[List[int], int], median_points: List[np.ndarray]
):
    if isinstance(labels, int):
        mask = np.argwhere(seg_data[labels])
        if mask.size == 0:
            return median_points
        median = np.median(mask, axis=0)
    elif isinstance(labels, list):
        label_left, label_right = labels
        pts_left = np.argwhere(seg_data[label_left])
        pts_right = np.argwhere(seg_data[label_right])
        if pts_left.size == 0 or pts_right.size == 0:
            return median_points
        median_left = np.median(pts_left, axis=0)
        median_right = np.median(pts_right, axis=0)
        median = np.mean([median_left, median_right], axis=0)
    else:
        raise ValueError("labels should be int or list of int")

    median_points.append(median)
    return median_points


def cleanup(seg_masks: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Removes components of specified organs (liver, spleen, stomach) that lie above a reference Z-bound,
    defined by the top slice of liver and lungs. Also removes small leftover components.

    The function processes these organ labels:
    - Liver (14), Spleen (24), Stomach (25)

    Args:
        seg_masks (dict[int, np.ndarray]): Dictionary mapping class labels to binary segmentation masks.

    Returns:
        dict[int, np.ndarray]: Updated segmentation dictionary with cleaned-up components.
    """
    target_labels = [14, 24, 25]  # liver, spleen, stomach
    ref_labels = [14, 15, 16]  # liver, left lung, right lung
    ref_coords = np.argwhere(
        np.logical_or.reduce([seg_masks.get(lbl, 0) for lbl in ref_labels])
    )
    z_upper_bound = ref_coords[:, 2].max() if ref_coords.size > 0 else np.inf

    for label in target_labels:
        if np.sum(seg_masks.get(label, 0)) < 1:
            continue

        labeled_comp, num_comps = binary_connected_components(seg_masks[label])
        removed_count = 0

        for i in range(1, num_comps + 1):
            comp_mask = labeled_comp == i
            comp_coords = np.argwhere(comp_mask)
            comp_mean_z = comp_coords[:, 2].mean()

            if comp_mean_z > z_upper_bound:
                seg_masks[label][comp_mask] = 0
                removed_count += 1

        seg_masks = remove_small_components(seg_masks, label, min_size=30)

    return seg_masks


def split_left_right(seg_masks: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Splits ambiguous bilateral organ labels into left/right based on spatial location.

    The function uses a reference midline plane estimated from known midline organs
    (e.g., aorta, postcava, adrenal glands, lungs, kidneys). It uses the liver position
    to determine the body's left-right orientation. Connected components are then
    reassigned to the appropriate left or right label depending on which side of the plane
    their centroid lies.

    Args:
        seg_masks (dict[int, np.ndarray]): Dictionary mapping organ class labels to binary masks.

    Returns:
        dict[int, np.ndarray]: Updated segmentation masks with left/right labels reassigned appropriately.
    """
    # Estimate midline from midline organs
    midline_refs = [
        [3, 22],  # aorta, postcava
        [1, 2],  # adrenal glands
        [15, 16],  # lungs
        [12, 13],  # kidneys
    ]

    median_points = []
    for ref_labels in midline_refs:
        median_points = calculate_median(seg_masks, ref_labels, median_points)

    # Define the midline plane
    point_on_plane = median_points[0]
    normal_vector = np.array(
        [-1.0, 0.0, 0.0]
    )  # Plane is orthogonal to x-axis (left/right)

    # Determine left/right direction based on liver location
    liver_median = calculate_median(seg_masks, 14, [])[0]
    liver_side = np.sign(np.dot(liver_median - point_on_plane, normal_vector)).astype(
        int
    )
    right_side = liver_side
    left_side = -liver_side

    # Reassign components to left/right based on centroid
    for left_label, right_label in relevant_left_right_pairs.values():
        for current_label, opposite_label, expected_side in [
            (left_label, right_label, left_side),
            (right_label, left_label, right_side),
        ]:
            labeled_comps, num_comps = binary_connected_components(
                seg_masks[current_label]
            )

            for comp_id in range(1, num_comps + 1):
                comp_coords = np.argwhere(labeled_comps == comp_id)
                centroid = comp_coords.mean(axis=0)
                side = int(np.sign(np.dot(centroid - point_on_plane, normal_vector)))

                if side != expected_side:
                    # Move the component to the opposite label
                    seg_masks[current_label][tuple(comp_coords.T)] = 0
                    seg_masks[opposite_label][tuple(comp_coords.T)] = 1

    return seg_masks


def suppress_non_largest_components(
    seg_masks: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """
    Suppress all but the largest connected component for each organ mask, unless the organ
    is allowed to be fragmented.

    Args:
        seg_masks (dict[int, np.ndarray]): Dictionary mapping class labels to binary segmentation masks.

    Returns:
        dict[int, np.ndarray]: The updated segmentation dictionary with only the largest component
                               kept for each non-fragmented label.
    """
    logger.debug("Suppressing non-largest connected components")

    for label, mask in seg_masks.items():
        if mask.sum() == 0:
            continue

        # Skip labels that are allowed to remain fragmented
        if label not in keep_fragmented_labels.values():
            labeled_comp = binary_connected_components(mask)[0]
            comp_sizes = np.bincount(labeled_comp.ravel())
            comp_sizes[0] = 0  # Ignore background
            largest_comp_label = comp_sizes.argmax()

            # Keep only the largest component
            seg_masks[label] = labeled_comp == largest_comp_label

    return seg_masks


def apply_binary_closing(
    seg_masks: Dict[int, np.ndarray],
    target_label: int,
    connectivity: int = 1,
    iterations: int = 1,
) -> Dict[int, np.ndarray]:
    """
    Applies binary morphological closing to a specific label in the segmentation mask.

    Closing helps fill small holes and bridge narrow gaps in the binary mask.

    Args:
        seg_masks (dict[int, np.ndarray]): Dictionary mapping organ labels to binary masks.
        target_label (int): The label whose mask should be processed.
        connectivity (int, optional): Defines connectivity for the structuring element (1 = 6-connectivity, 2 = 18, 3 = 26). Default is 1.
        iterations (int, optional): Number of times the closing operation is applied. Default is 1.

    Returns:
        dict[int, np.ndarray]: Updated segmentation mask dictionary with closing applied to the target label.
    """
    seg_masks[target_label] = binary_closing(
        seg_masks[target_label],
        structure=generate_binary_structure(3, connectivity),
        iterations=iterations,
    )
    return seg_masks


def remove_small_components(
    seg_masks: Dict[int, np.ndarray], target_label: int, min_size: int = 10
) -> Dict[int, np.ndarray]:
    """
    Removes small connected components from the binary mask corresponding to the given target label.

    Args:
        seg_masks (dict[int, np.ndarray]): A dictionary mapping class labels to binary segmentation masks.
        target_label (int): The label of the class to process.
        min_size (int, optional): Minimum number of pixels required for a component to be retained. Defaults to 10.

    Returns:
        dict[int, np.ndarray]: The updated segmentation mask dictionary with small components removed from the target label's mask.
    """
    labeled_comps, num_comps = binary_connected_components(seg_masks[target_label])

    for i in range(1, num_comps + 1):
        comp_mask = labeled_comps == i
        if np.sum(comp_mask) < min_size:
            seg_masks[target_label][comp_mask] = 0

    return seg_masks


def process_organs(data_inputs):
    """
    Assumtion: liver, spleen, stomach, kidneys, aorta/postcava must exist in the class map.
    Flow chart:
    aorta/postcava -> left-right labels -> other organs -> supress non-largest components
    """
    name2label_supported = {name: label for label, name in class_maps_supported.items()}

    source_dir, target_dir, save_combined_labels = data_inputs
    logger.debug(f"Processing {source_dir.name}")

    seg_files = list((source_dir / "segmentations").glob("*.nii.gz"))
    seg_masks = {}
    files_skipped = set()
    for f in seg_files:
        organ_name = f.name.split(".", 1)[0]
        if organ_name in name2label_supported:  # supported labels
            # TODO: implement lazy loading
            seg_masks[name2label_supported[organ_name]] = (
                nib.load(f).get_fdata().astype(bool)
            )
        else:
            logger.debug(f"Organ {organ_name} not supported.")
            files_skipped.add(f)
    if len(seg_masks) == 0:
        logger.error(
            f"No valid segmentation files found in {source_dir.name}. Skipping."
        )
        return

    # standarize orientation
    logger.debug("Standardizing orientation")
    example_seg = nib.load(seg_files[0])
    affine = example_seg.affine
    header = example_seg.header
    orig_ornt = io_orientation(affine)
    intermediate_ornt = axcodes2ornt(intermediate_axcodes)
    transform = ornt_transform(orig_ornt, intermediate_ornt)
    for label, data in seg_masks.items():
        seg_masks[label] = apply_orientation(data, transform)

    mask_cropper = MaskCropper()
    mask_cropper.get_tightest_bbox(seg_masks)
    seg_masks = mask_cropper.crop(seg_masks)

    # make sure lung postprocessing follows liver postprocessing
    processing_sqeuence = [
        3,  # aorta
        22,  # postcava
        1,  # adrenal gland left
        2,  # adrenal gland right
        12,  # kidney left
        13,  # kidney right
        24,  # spleen
        25,  # stomach
        14,  # liver
        11,  # gall bladder
        15,  # lung left
        16,  # lung right
        4,  # bladder
        23,  # prostate
        9,  # femur left
        10,  # femur right
        18,  # pancreas body
        19,  # pancreas head
        20,  # pancreas tail
        17,  # pancreas
        21,  # pancreatic duct
        5,  # artery
        6,  # colon
        7,  # common bile duct
        8,  # duodenum
        26,  # superior_mesenteric_artery
        27,  # veins
        29,  # esophagus
        31,  # intestine
    ]

    # Initialize the organ rectifier
    rectifier = OrganRectifier()

    class_list = list(seg_masks.keys())
    class_list.sort(key=lambda x: processing_sqeuence.index(x))

    logger.debug(f"Processing classes {class_list}")
    # process aorta and postcava first
    assert (
        3 in class_list and 22 in class_list
    ), f"Aorta/postcava not found in {class_list}"
    logger.debug(f"Processing aorta/postcava")
    for label in [3, 22]:
        seg_masks = rectifier.remove_out_of_bounds_components(seg_masks, label)
        class_list.remove(label)

    # process liver
    assert 14 in class_list, f"Liver not found in {class_list}"
    logger.debug(f"Processing liver")
    label = 14
    seg_masks = rectifier.remove_out_of_bounds_components(seg_masks, label)
    class_list.remove(label)

    logger.debug(f"Processing paired organs")
    seg_masks = split_left_right(seg_masks)

    # deal with the rest of the organs
    logger.debug("Processing other organs")
    for label in class_list:
        try:
            # remove out of bounds components
            if label in [4, 23]:
                seg_masks = rectifier.apply_bladder_or_prostate_postprocessing(
                    seg_masks, label
                )
            elif label in [15, 16]:
                seg_masks = rectifier.apply_lung_postprocessing(seg_masks, label)
            elif label == 21:
                seg_masks = rectifier.apply_pancreatic_duct_post_processing(
                    seg_masks, label
                )
            else:
                seg_masks = rectifier.remove_out_of_bounds_components(seg_masks, label)

            # reassignment
            if label == 6:  # colon
                seg_masks = rectifier.reassign_components(
                    seg_masks, label, reassignment_candidates=[15, 16, 26, 31]
                )
            elif label == 8:  # duodenum
                seg_masks = rectifier.reassign_components(
                    seg_masks, label, reassignment_candidates=[24, 25]
                )
            elif label == 12:  # left kidney
                seg_masks = rectifier.reassign_components(
                    seg_masks, label, reassignment_candidates=[24, 25]
                )
            elif label == 13:  # right kidney
                seg_masks = rectifier.reassign_components(
                    seg_masks, label, reassignment_candidates=[11, 14, 25]
                )
            elif label in [14, 17, 24]:  # liver, pancreas, spleen
                seg_masks = rectifier.reassign_components(
                    seg_masks,
                    label,
                    reassignment_candidates=[12, 13, 14, 24, 25],
                )
            elif label in [18, 19, 20]:  # pancreas body, head, tail
                seg_masks = rectifier.reassign_components(
                    seg_masks, label, reassignment_candidates=[18, 19, 20]
                )
            elif label == 31:  # intestine
                seg_masks = rectifier.reassign_components(
                    seg_masks, label, reassignment_candidates=[6, 14, 24, 25]
                )
        except Exception as e:
            import traceback

            logger.error(f"Got an error when processing file {source_dir.name}")
            logger.error(e)
            logger.error(traceback.format_exc())
    seg_masks = cleanup(seg_masks)
    seg_masks = suppress_non_largest_components(seg_masks)
    for label, mask in seg_masks.items():
        # if label in keep_fragmented_labels.values():
        seg_masks = remove_small_components(seg_masks, label)

    seg_masks = mask_cropper.restore(seg_masks)

    # tranform back to original orientation
    inverse_transform = ornt_transform(intermediate_ornt, orig_ornt)
    for label, data in seg_masks.items():
        seg_masks[label] = apply_orientation(data.astype(np.uint8), inverse_transform)

    ################### Save the results ###################
    ind_target_dir = target_dir / "segmentations"
    ind_target_dir.mkdir(parents=True, exist_ok=True)

    # save individual labels
    for f in seg_files:
        ind_target_path = ind_target_dir / f.name
        corresponding_label = name2label_supported.get(f.name.split(".", 1)[0], None)
        if corresponding_label is not None:
            ind_seg_mask = seg_masks[corresponding_label]

            ind_seg = nib.Nifti1Image(
                ind_seg_mask.astype(np.uint8),
                affine,
                header,
            )
            nib.save(ind_seg, ind_target_path)
            del ind_seg_mask, ind_seg
        else:
            shutil.copy(f, ind_target_path)

    # save combined labels
    if save_combined_labels:
        combined_seg = np.zeros(example_seg.shape, dtype=np.uint8)
        class_maps_output = {}
        unsupported_label_index = max(class_maps_supported) + 1
        for f in files_skipped:
            mask = nib.load(f).get_fdata().astype(bool)
            combined_seg[mask.astype(bool)] = unsupported_label_index
            class_maps_output[unsupported_label_index] = f.name.split(".", 1)[0]
            unsupported_label_index += 1
        for label, mask in sorted(seg_masks.items()):
            combined_seg[mask.astype(bool)] = label
            class_maps_output[label] = class_maps_supported[label]

        combined_nifti = nib.Nifti1Image(combined_seg, affine, header)
        combined_path = target_dir / "combined_labels.nii.gz"
        nib.save(combined_nifti, combined_path)

        # save new class maps
        class_maps_output = dict(sorted(class_maps_output.items()))
        with open(target_dir / "class_maps.json", "w") as f:
            json.dump(class_maps_output, f, indent=4)

    # copy CT image if exists
    if (source_dir / "ct.nii.gz").exists():
        shutil.copy(source_dir / "ct.nii.gz", target_dir / "ct.nii.gz")

    logger.debug(f"Finished processing {source_dir.name}")


if __name__ == "__main__":
    args = auto_cli(Settings)
    source_dir = Path(args.input_folder)
    target_dir = Path(args.output_folder)
    cpu_count = args.cpu_count

    source_sub_dirs = sorted(
        [i for i in source_dir.iterdir() if i.name.startswith("BDMAP")],
        key=lambda x: x.name,
    )
    if args.end_idx == -1:
        args.end_idx = len(source_sub_dirs)
    source_sub_dirs = source_sub_dirs[args.start_idx : args.end_idx]

    logger.info(
        f"Processing {len(source_sub_dirs)} source folders from index {args.start_idx} to {args.end_idx or 'end'}"
    )

    data_inputs = [
        (source_sub_dir, target_dir / source_sub_dir.name, args.save_combined_labels)
        for source_sub_dir in source_sub_dirs
    ]

    if cpu_count > 1:
        cpu_count = min(cpu_count, len(data_inputs))
        logger.info(f"Start post-processing with {cpu_count} jobs")
        logger_initializer = partial(init_logger, verbose=args.verbose)
        with multiprocessing.Pool(cpu_count, initializer=logger_initializer) as pool:
            results = pool.imap_unordered(process_organs, data_inputs)
            for _ in tqdm(results, total=len(data_inputs), desc="Processing"):
                pass  # We only care about the progress
    else:
        logger.warning(
            "This is for debugging purpose. Please use multiprocessing (i.e., set cpu_count > 1) if you are processing a large number of files."
        )
        for i in data_inputs:
            if "BDMAP_00000001" in i[0].name:
                from time import time

                start = time()
                process_organs(i)
                end = time()
                logger.info(
                    f"Processing time for {i[0].name}: {end - start:.2f} seconds"
                )
                break
    logger.info("Finish post-processing")
