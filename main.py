import argparse
import multiprocessing
from multiprocessing import cpu_count
from organs_postprocessing import *
from tqdm import tqdm
import logging
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
subfolder_name = config['subfolder_name']
affine_reference_file_name = os.path.join(subfolder_name, config['affine_reference_file_name'])
target_organs = set(config.get('target_organs', []))


reference_file_name =  affine_reference_file_name # affine info
data_type = np.int16

# set up logging 
logging.basicConfig(
    filename='postprocessing_errors.log',  
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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


def process_organs(segmentation_dict: dict, reference_img, combined_seg: np.array, target_organs: set):
    """
    Apply organ-specific post-processing functions to the segmentation dict
    based on user-defined target organs.
    """
    axis_map = get_axis_map(reference_img)
    segmentation_dict = reassign_false_positives(segmentation_dict, organ_adjacency_map)

    # Apply processing only if target organ is in the list
    if 'stomach' in target_organs:
        segmentation_dict = post_processing_stomach(segmentation_dict)
    if 'liver' in target_organs:
        segmentation_dict = post_processing_liver(segmentation_dict)
    if 'pancreas' in target_organs:
        segmentation_dict = post_processing_pancreas(segmentation_dict)
    if 'colon' in target_organs or 'intestine' in target_organs:
        segmentation_dict = post_processing_colon_intestine(segmentation_dict)
    if 'spleen' in target_organs:
        segmentation_dict = post_processing_spleen(segmentation_dict)
    if 'duodenum' in target_organs:
        segmentation_dict = post_processing_duodenum(segmentation_dict)

    calibration_standards_mask = segmentation_dict.get('liver')

    if 'lung' in target_organs:
        segmentation_dict = post_processing_lung(segmentation_dict, axis_map, calibration_standards_mask)
    if 'kidney' in target_organs:
        segmentation_dict = post_processing_kidney(segmentation_dict, axis_map, calibration_standards_mask)
    if 'femur' in target_organs:
        segmentation_dict = post_processing_femur(segmentation_dict, axis_map, calibration_standards_mask)
    if 'adrenal_gland' in target_organs:
        segmentation_dict = post_processing_adrenal_gland(segmentation_dict, axis_map, calibration_standards_mask)

    if 'aorta' in target_organs or 'postcava' in target_organs:
        segmentation_dict = post_processing_aorta_postcava(segmentation_dict)
    if 'bladder' in target_organs or 'prostate' in target_organs:
        segmentation_dict = post_processing_bladder_prostate(
            segmentation_dict,
            segmentation=combined_seg,
            axis=axis_map['z']
        )

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
        folder_path=input_path,
        subfolder_name=subfolder_name
    )

    # combine later as calibration reference
    segmentation = combine_segmentation_dict(segmentation_dict, class_map)

    # process
    postprocessed_segmentation_dict = process_organs(
        segmentation_dict, 
        img,
        segmentation,
        target_organs)
    
    # save
    save_folder_path = os.path.join(output_path, input_folder_name)
    os.makedirs(save_folder_path, exist_ok=True)

    save_and_combine_segmentations(
        processed_segmentation_dict=postprocessed_segmentation_dict,
        class_map=class_map,
        reference_img=img,
        output_folder=save_folder_path,
        if_save_combined=False
    )


############################## Parallel Execution ################################
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError, CancelledError

def process_case(sub_folder, input_folder, output_folder):
    input_path = os.path.join(input_folder, sub_folder)
    print(f"[INFO] Processing {sub_folder}")
    main(
        input_path=input_path,
        input_folder_name=sub_folder,
        output_path=output_folder,
    )


def run_in_parallel(sub_folders, input_folder, output_folder, max_workers=4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_case, sub_folder, input_folder, output_folder): sub_folder
            for sub_folder in sub_folders
        }

        for future in tqdm(as_completed(futures), total=len(futures), ncols=66, desc="Processing cases"):
            sub_folder = futures[future]
            try:
                future.result()
            except MemoryError as mem_err:
                logging.error(f"MemoryError while processing {sub_folder}: {mem_err}")
                print(f"[WARNING] MemoryError in {sub_folder}, skipping.")
                
            except Exception as e:
                logging.error(f"Exception(Not MemError) while processing {sub_folder}: {e}")
                print(f"[WARNING] Error(Not MemError)  in {sub_folder}, skipping.")


parser = argparse.ArgumentParser(description="Anatomical-aware post-processing")
parser.add_argument('--input_folder', type=str, help='Input files folder location, /path/to/input/data')
parser.add_argument('--output_folder', type=str, help='Output files folder location, /path/to/save/results')
parser.add_argument('--cpu_count', type=int, default=cpu_count(), help='Number of CPU cores to use for parallel processing (default: system max)')
args = parser.parse_args()



if __name__ == '__main__':

    input_folder = args.input_folder
    output_folder= args.output_folder
    sub_folders = [sf for sf in os.listdir(input_folder) if sf != '.DS_Store']
    sub_folders.sort()

    # safe execution without freezing the system
    max_workers = min(args.cpu_count, len(sub_folders), multiprocessing.cpu_count() - 1)
    print(f"[INFO] Starting... with {max_workers} multiprocess ...\n\n")
    
    print(f"[INFO] Input files dir: {input_folder}\n[INFO] Output files dir: {output_folder}")
    run_in_parallel(sub_folders, input_folder, output_folder, max_workers=max_workers)
