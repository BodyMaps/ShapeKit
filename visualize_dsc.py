from organs_postprocessing import *
from tqdm import tqdm
from config import affine_reference_file_name
import pandas as pd
import seaborn as sns




reference_file_name =  affine_reference_file_name # affine info
data_type = np.int16

def combine_segmentation_dict(segmentation_dict: dict, class_map: dict) -> np.ndarray:
    """
    Combine organ segmentations
    """
    shape = next(iter(segmentation_dict.values())).shape
    combined = np.zeros(shape, dtype=np.uint8)

    for index, organ_name in class_map.items():
        mask = segmentation_dict.get(organ_name)
        if mask is None or np.sum(mask) == 0:
            continue
        combined[mask > 0] = index

    return combined


def process_organs(segmentation_dict:dict, reference_img, combined_seg:np.array):
    """
    Apply organ-specific post-processing functions to the segmentation dict.
    Organs not listed in defined functions will be ignored.
    """

    # get the affine axis
    axis_map = get_axis_map(reference_img)

    # using copy to avoid global change
    segmentation_dict = segmentation_dict.copy()
    # deal with fp
    
    segmentation_dict = reassign_FalsePositives(segmentation_dict, organ_adjacency_map)
    
    # larger organs piror to smaller ones
    segmentation_dict = post_processing_stomach(segmentation_dict)
    segmentation_dict = post_processing_liver(segmentation_dict)

    # for calibration to define which one shall be on the right side
    calibration_standards_mask = segmentation_dict.get('liver')

    segmentation_dict = post_processing_colon(segmentation_dict)
    segmentation_dict = post_processing_pancreas(segmentation_dict)
    segmentation_dict = post_processing_spleen(segmentation_dict)


    segmentation_dict = post_processing_lung(segmentation_dict, axis_map, calibration_standards_mask)
    segmentation_dict = post_processing_colon(segmentation_dict) # double check with colon so that to ensure removing all the redundant part
    segmentation_dict = post_processing_kidney(segmentation_dict, axis_map, calibration_standards_mask)
    segmentation_dict = post_processing_femur(segmentation_dict, axis_map, calibration_standards_mask)
    segmentation_dict = post_processing_adrenal_gland(segmentation_dict, axis_map, calibration_standards_mask)

    segmentation_dict = post_processing_aorta_postcava(segmentation_dict)
    segmentation_dict = post_processing_bladder_prostate(segmentation_dict, # use the combined seg for calibration
                                                         segmentation= combined_seg,
                                                         axis=axis_map['z'])
    
    return segmentation_dict



def dice_score(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    size_sum = mask1.sum() + mask2.sum()
    if size_sum == 0:
        return 1.0  # Both masks are empty
    return 2. * intersection / size_sum


def calculate_dsc_all_organs(gt_dict, pred_dict):
    """
    Calculate DSC for each organ (not None) and overall DSC.
    """
    dsc_per_organ = {}
    gt_all = None
    pred_all = None

    
    for organ in gt_dict:
        gt_mask = gt_dict[organ]
        pred_mask = pred_dict.get(organ)
        if gt_mask is not None and pred_mask is not None:

            dsc = dice_score(gt_mask, pred_mask)
            if dsc < 0.1:# too small, which means shall be removed
                dsc = 1.0
            dsc_per_organ[organ] = dsc

            # Union masks for overall DSC

            if gt_all is None:
                gt_all = gt_mask.astype(bool)
                pred_all = pred_mask.astype(bool)
            else:
                gt_all = np.logical_or(gt_all, gt_mask.astype(bool))
                pred_all = np.logical_or(pred_all, pred_mask.astype(bool))
        
        if pred_mask is None:
            # shall not exist
            print(f"[INFO] {organ} should not exist in real scene")
            dsc_per_organ[organ] = 1.0

                
    # Calculate overall DSC
    overall_dsc = dice_score(gt_all, pred_all) if (gt_all is not None and pred_all is not None) else None
    return dsc_per_organ, overall_dsc



if __name__ == '__main__':
    input_folder = 'T'
    output_folder = 'NT'
    sub_folders = [sf for sf in os.listdir(input_folder) if sf != '.DS_Store']


    def new_main(input_path, input_folder_name, output_path=None):
        """
        input_path: the folder path
        
        """
        
        # get the reference img
        seg_path = os.path.join(input_path, reference_file_name)
        img = nib.load(seg_path)
        
        # load segmentations
        segmentation_dict = read_all_segmentations(
            folder_path=input_path
        )

        # combine later as calibration reference
        segmentation = combine_segmentation_dict(segmentation_dict, class_map)

        # process
        postprocessed_segmentation_dict = process_organs(
            segmentation_dict, 
            img,
            segmentation)
        
        dsc_per_organ, overall_dsc = calculate_dsc_all_organs(segmentation_dict, postprocessed_segmentation_dict)


        save_folder_path = os.path.join(output_path, input_folder_name)
        os.makedirs(save_folder_path, exist_ok=True)

        save_and_combine_segmentations(
            processed_segmentation_dict=postprocessed_segmentation_dict,
            class_map=class_map,
            reference_img=img,
            output_folder=save_folder_path
        )

        return dsc_per_organ, overall_dsc


    from tqdm import tqdm

    organ_dsc_dict = {}  # organ -> list of (subfolder, dsc)

    for sub_folder in tqdm(sub_folders):
        input_path = os.path.join(input_folder, sub_folder)
        print(f"[INFO] Processing {sub_folder}")

        dsc_per_organ, overall_dsc = new_main(input_path, sub_folder, output_folder)
        print(f"[INFO] {sub_folder} average dsc: {overall_dsc:.2f}")

        for organ, dsc in dsc_per_organ.items():
            if organ not in organ_dsc_dict:
                organ_dsc_dict[organ] = []
            organ_dsc_dict[organ].append((sub_folder, dsc))

    # Compute mean DSC per organ
    organ_mean_dsc = {
        organ: np.mean([dsc for _, dsc in dsc_list])
        for organ, dsc_list in organ_dsc_dict.items()
    }

    # Find the 5 organs with lowest mean DSC
    lowest_organs = sorted(organ_mean_dsc, key=organ_mean_dsc.get)[:7]

    # Report the worst-case subfolder for each of those 5 organs
    print("\n\n5 organs with lowest mean DSC:")
    for organ in lowest_organs:
        worst_case = min(organ_dsc_dict[organ], key=lambda x: x[1])
        print(f"{organ} lowest: {worst_case[0]} (DSC: {worst_case[1]:.2f})")




    # make the violin plot
    plot_data = []
    for organ in lowest_organs:
        for sub_folder, dsc_value in organ_dsc_dict[organ]:
            plot_data.append({'Organ': organ, 'Subfolder': sub_folder, 'DSC': dsc_value})

    df = pd.DataFrame(plot_data)

    # 5. Plot violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Organ', y='DSC', data=df, inner=None, color="skyblue", cut=0)
    sns.pointplot(x='Organ', y='DSC', data=df, linestyle='none', color="k", markers="D", capsize=.1, err_kws={'linewidth': 1})  # Mean+std


    plt.title("DSC scores for organs with lowest average DSC")
    plt.ylabel("DSC")
    plt.xlabel("Organ")
    plt.tight_layout()
    plt.show()
