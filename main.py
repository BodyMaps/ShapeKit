import argparse
from multiprocessing import cpu_count
from organs_postprocessing import *
from tqdm import tqdm
from config import affine_reference_file_name


reference_file_name =  affine_reference_file_name # affine info
data_type = np.int16
##############################################################


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

    # deal with fp
    segmentation_dict = reassign_FalsePositives(segmentation_dict, organ_adjacency_map)

    # larger organs piror to smaller ones
    segmentation_dict = post_processing_stomach(segmentation_dict)
    segmentation_dict = post_processing_liver(segmentation_dict)
    segmentation_dict = post_processing_pancreas(segmentation_dict)
    segmentation_dict = post_processing_colon(segmentation_dict)
    segmentation_dict = post_processing_spleen(segmentation_dict)
    
    # for calibration to define which one shall be on the right side
    calibration_standards_mask = segmentation_dict.get('liver')
    
    segmentation_dict = post_processing_lung(segmentation_dict, axis_map, calibration_standards_mask)
    segmentation_dict = post_processing_kidney(segmentation_dict, axis_map, calibration_standards_mask)
    segmentation_dict = post_processing_femur(segmentation_dict, axis_map, calibration_standards_mask)
    segmentation_dict = post_processing_adrenal_gland(segmentation_dict, axis_map, calibration_standards_mask)

    segmentation_dict = post_processing_aorta_postcava(segmentation_dict)
    segmentation_dict = post_processing_bladder_prostate(segmentation_dict, # use the combined seg for calibration
                                                         segmentation= combined_seg,
                                                         axis=axis_map['z'])
    
    return segmentation_dict



def main(input_path, input_folder_name, output_path=None):
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
    
    # save
    save_folder_path = os.path.join(output_path, input_folder_name)
    os.makedirs(save_folder_path, exist_ok=True)

    save_and_combine_segmentations(
        processed_segmentation_dict=postprocessed_segmentation_dict,
        class_map=class_map,
        reference_img=img,
        output_folder=save_folder_path
    )


############################## Parallel Execution ################################
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_case(sub_folder, input_folder, output_folder):
    input_path = os.path.join(input_folder, sub_folder)
    print(f"[INFO] Processing {sub_folder}")
    main(
        input_path=input_path,
        input_folder_name=sub_folder,
        output_path=output_folder
    )

def run_in_parallel(sub_folders, input_folder, output_folder, max_workers=4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_case, sub_folder, input_folder, output_folder)
            for sub_folder in sub_folders
        ]
        for future in tqdm(as_completed(futures), total=len(futures), ncols=66, desc="Processing cases"):
            
            future.result()


parser = argparse.ArgumentParser(description="Anatomical-aware post-processing")
parser.add_argument('--input_folder', type=str, help='Input files folder location, /path/to/input/data')
parser.add_argument('--output_folder', type=str, help='Output files folder location, /path/to/save/results')
parser.add_argument('--cpu_count', type=int, default=cpu_count(), help='Number of CPU cores to use for parallel processing (default: system max)')
args = parser.parse_args()


print("[INFO] Parsed arguments:")
for k, v in vars(args).items():
    print(f"  {k}: {v}")

if __name__ == '__main__':

    input_folder = args.input_folder
    output_folder= args.output_folder
    sub_folders = [sf for sf in os.listdir(input_folder) if sf != '.DS_Store']
    
    print(f"[INFO] Input files dir: {input_folder}\n[INFO] Output files dir: {output_folder}")
    run_in_parallel(sub_folders, input_folder, output_folder, max_workers=args.cpu_count)
