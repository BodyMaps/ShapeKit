import argparse
import multiprocessing
from multiprocessing import cpu_count
from utils.organs_postprocessing import *
from utils.vertebrae_postprocessing import postprocessing_vertebrae
import logging
import yaml
import traceback

from multiprocessing import Pool
from tqdm import tqdm
import logging
import time
import csv

import shutil



"""
mail: jliu452@uw.edu
last systematic update: Dec 23, 2025

"""
##############################################################
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
subfolder_name = config['subfolder_name']
affine_reference_file_name = os.path.join(subfolder_name, config['affine_reference_file_name'])
target_organs = set(config.get('target_organs', []))
organ_list = list(class_map.values())
reference_file_name =  affine_reference_file_name # affine info
data_type = np.int16
save_combined_label_bool = bool(config['if_save_combined_label'])

##############################################################



def check_unprocessed_cases(input_folder: str, output_folder: str, csv_path: str = "continue.csv"):
    """
    Continue-prediction module
    """
    input_patients = sorted(
        [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    )

    unprocessed = []

    for pid in input_patients:
        out_dir = os.path.join(output_folder, pid)
        seg_dir = os.path.join(out_dir, "segmentations")

        processed = (
            os.path.isdir(seg_dir)
            and len(os.listdir(seg_dir)) > 0
        )

        if not processed:
            unprocessed.append(pid)

    # write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Inference ID"])
        for pid in unprocessed:
            writer.writerow([pid])

    print(f"[INFO] Found {len(unprocessed)} unprocessed cases. Saved to {csv_path}")

    return unprocessed




def combine_segmentation_dict(segmentation_dict: dict, class_map: dict) -> np.ndarray:
    """
    Combine organ segmentations
    """
    shape = next(iter(segmentation_dict.values())).shape
    combined = np.zeros(shape, dtype=np.uint8)

    for index, organ_name in class_map.items():
        mask = segmentation_dict.get(organ_name)
        if mask is None or not np.any(mask):
            continue
        combined[mask > 0] = index

    return combined


def process_organs(segmentation_dict: dict, reference_img, combined_seg: np.array, target_organs: set, patient_id: str, logger: logging.Logger,
):
    """
    Apply organ-specific post-processing functions to the segmentation dict
    based on user-defined target organs.
    """
    axis_map = get_axis_map(reference_img)

    segmentation_dict = reassign_false_positives(
        segmentation_dict,
        organ_adjacency_map,
        patient_id=patient_id,
        logger=logger,
    )

    if 'stomach' in target_organs:
        segmentation_dict = post_processing_stomach(segmentation_dict)

    if 'liver' in target_organs:
        segmentation_dict = post_processing_liver(segmentation_dict)

    if 'pancreas' in target_organs:
        segmentation_dict = post_processing_pancreas(segmentation_dict)

    if 'colon' in target_organs or 'intestine' in target_organs:
        segmentation_dict = post_processing_colon_intestine(
            segmentation_dict,
            patient_id=patient_id,
            logger=logger,
        )

    if 'spleen' in target_organs:
        segmentation_dict = post_processing_spleen(segmentation_dict)

    if 'duodenum' in target_organs:
        segmentation_dict = post_processing_duodenum(segmentation_dict)

    calibration_standards_mask = segmentation_dict.get('liver')

    if 'lung' in target_organs:
        segmentation_dict = post_processing_lung(
            segmentation_dict,
            axis_map,
            calibration_standards_mask,
            patient_id=patient_id,
            logger=logger,
        )

    if 'kidney' in target_organs:
        segmentation_dict = post_processing_kidney(
            segmentation_dict,
            axis_map,
            calibration_standards_mask,
            patient_id=patient_id,
            logger=logger,
        )

    if 'femur' in target_organs:
        segmentation_dict = post_processing_femur(
            segmentation_dict,
            axis_map,
            calibration_standards_mask,
            patient_id=patient_id,
            logger=logger,
        )

    if 'adrenal_gland' in target_organs:
        segmentation_dict = post_processing_adrenal_gland(
            segmentation_dict,
            axis_map,
            calibration_standards_mask,
        )

    if 'aorta' in target_organs or 'postcava' in target_organs:
        segmentation_dict = post_processing_aorta_postcava(segmentation_dict)

    if 'bladder' in target_organs or 'prostate' in target_organs:
        segmentation_dict = post_processing_bladder_prostate(
            segmentation_dict,
            segmentation=combined_seg,
            axis=axis_map['z'],
            patient_id=patient_id,
            logger=logger,
        )

    if 'vertebrae' in target_organs:
        segmentation_dict = postprocessing_vertebrae(
            patient_id,
            segmentation_dict,
            logger=logger,
        )

    return segmentation_dict



def main(input_path, input_folder_name, output_path=None):
    """
    input_path: the folder path
    """
    
    # ---- NEW: copy input patient folder to output first ----
    if output_path is not None:
        dst_path = os.path.join(output_path, input_folder_name)
        if not os.path.exists(dst_path):
            shutil.copytree(input_path, dst_path)
        else:
            pass
    # --------------------------------------------------------
    
    seg_path = os.path.join(input_path, reference_file_name)
    img = nib.load(seg_path)
    
    segmentation_dict = read_all_segmentations(
        folder_path=input_path,
        organ_list=organ_list,
        subfolder_name=subfolder_name
    )

    segmentation = combine_segmentation_dict(segmentation_dict, class_map)
    patient_id = os.path.basename(input_path)

    postprocessed_segmentation_dict = process_organs(
        segmentation_dict, 
        img,
        segmentation,
        target_organs,
        patient_id = patient_id,
        logger = logging,
    )
    
    save_folder_path = os.path.join(output_path, input_folder_name)
    os.makedirs(save_folder_path, exist_ok=True)

    save_and_combine_segmentations(
        processed_segmentation_dict=postprocessed_segmentation_dict,
        class_map=class_map,
        reference_img=img,
        output_folder=save_folder_path,
        if_save_combined=save_combined_label_bool
    )

    del img
    del segmentation_dict
    del segmentation
    del postprocessed_segmentation_dict
    gc.collect()



############################## Parallel Execution with multiprocessing.Pool ##############################
from multiprocessing import Pool

def process_case_wrapper(args):
    sub_folder, input_folder, output_folder = args
    try:
        # logging.info(f"[INFO] Processing {sub_folder}")
        input_path = os.path.join(input_folder, sub_folder)
        main(input_path, sub_folder, output_folder)
        post_logger.info(f"[ShapeKit] Successfully processed {sub_folder}")
    except MemoryError as mem_err:
        logging.error(f"MemoryError while processing {sub_folder}: {mem_err}")
    except Exception as e:
        logging.error(f"[CRASH] {sub_folder}: {e}")
        traceback.print_exc()


def run_in_parallel(sub_folders, input_folder, output_folder, max_workers=4, tqdm_ncols=80):
    logging.info(
        f"\n\n[INFO] Start processing {len(sub_folders)} cases with up to {max_workers} workers.\n\n"
    )

    args_list = [(sub_folder, input_folder, output_folder) for sub_folder in sub_folders]

    with Pool(processes=max_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_case_wrapper, args_list),
            total=len(args_list),
            desc="Processing cases",
            unit="case",
            ncols=tqdm_ncols,
        ):
            pass
############################## Parallel Execution with multiprocessing.Pool ##############################




parser = argparse.ArgumentParser(description="Anatomical-aware post-processing")
parser.add_argument('--input_folder', type=str, help='Input files folder location, /path/to/input/data')
parser.add_argument('--output_folder', type=str, help='Output files folder location, /path/to/save/results')
parser.add_argument('--log_folder', type=str, default='./logs/task_001', help='Logging folder location')
parser.add_argument('--csv', type=str, default=None, help='Guidence csv file telling ShapeKit specific ones for processing')
parser.add_argument('--cpu_count', type=int, default=cpu_count(), help='Number of CPU cores to use for parallel processing (default: system max)')
parser.add_argument('--continue_prediction', action="store_true", help='If continue from last processing record')
parser.add_argument('--tqdm_ncols', type=int, default=80, help='Width of tqdm progress bar in characters')
args = parser.parse_args()



# set up logging 
os.makedirs(args.log_folder, exist_ok=True)
logging.basicConfig(
    filename=f'{args.log_folder}/debug.log',  
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

post_logger = logging.getLogger("postprocessing")
post_logger.setLevel(logging.INFO)
post_handler = logging.FileHandler(f"{args.log_folder}/postprocessing.log")
post_handler.setFormatter(logging.Formatter(
    '[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
post_logger.propagate = False
post_logger.addHandler(post_handler)







if __name__ == '__main__':

    input_folder = args.input_folder
    output_folder= args.output_folder

    sub_folders = [sf for sf in os.listdir(input_folder) if sf != '.DS_Store']  
    sub_folders.sort()

    if args.continue_prediction:
        # generate continue.csv and only run those
        sub_folders_filtered = check_unprocessed_cases(
            input_folder=input_folder,
            output_folder=output_folder,
            csv_path="continue.csv",
        )

        len1 = len(sub_folders_filtered)
        len2 = len(sub_folders)
        print(f"[INFO] Resume from last prediction process, I will only work on {len1} out of total {len2} cases!")
        sub_folders = sub_folders_filtered
        sub_folders.sort()

    max_workers = min(args.cpu_count, len(sub_folders), multiprocessing.cpu_count() - 1)
    print(f"[INFO] Starting... with {max_workers} multiprocess ...")
    print(f"[INFO] Input files dir: {input_folder}")
    print(f"[INFO] Output files dir: {output_folder}")
    print(f"[INFO] Logging dir: {args.log_folder}")
    run_in_parallel(
        sub_folders,
        input_folder,
        output_folder,
        max_workers=max_workers,
        tqdm_ncols=args.tqdm_ncols,
    )
