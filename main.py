import argparse
import multiprocessing
from multiprocessing import cpu_count
from utils.organs_postprocessing import *
from utils.vertebrae_postprocessing import postprocessing_vertebrae
from utils.vertebrae_iterative import postprocessing_vertebrae as postprocessing_vertebrae_songlin
from utils.vertebrae_pro import postprocessing_vertebrae_pro
from utils.vertebrae_hao import process_case as process_hao_case
from utils.vertebrae_davir import postprocessing_vertebrae_davir
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
vertebrae_engine = config.get('vertebrae_engine', 'shapekit')
ct_file_name = config.get('ct_file_name', 'ct.nii.gz')
ct_root = config.get('ct_root', None)

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


def read_cases_from_csv(csv_path, key="Inference ID"):
    """
    Only the cases listed in the csv will be processed
    """
    cases = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if key not in reader.fieldnames:
            raise ValueError(f"[ERROR] CSV must contain column '{key}'")
        for row in reader:
            cases.append(row[key])
    return set(cases)





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
    ct_path: str = None,
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
        if vertebrae_engine == 'shapekit_pro':
            segmentation_dict = postprocessing_vertebrae_pro(
                patient_id,
                segmentation_dict,
                reference_img,
                ct_path,
                logger=logger,
            )
        elif vertebrae_engine == 'shapekit_davir':
            segmentation_dict = postprocessing_vertebrae_davir(
                patient_id,
                segmentation_dict,
                reference_img,
                ct_path,
                logger=logger,
            )
        elif vertebrae_engine == 'shapekit_songlin':
            segmentation_dict = postprocessing_vertebrae_songlin(
                patient_id,
                segmentation_dict,
                logger=logger,
            )
        else:
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
    
    if vertebrae_engine == 'shapekit_hao':
        # Isolate the evaluated algorithm from legacy organ cleanup and IO.
        # The strict adapter validates each mask/CT grid and writes zero masks.
        if target_organs != {'vertebrae'}:
            raise ValueError('shapekit_hao requires --vertebrae_only')
        ct_path = os.path.join(input_path, ct_file_name)
        if not os.path.exists(ct_path) and ct_root is not None:
            ct_path = os.path.join(ct_root, input_folder_name, ct_file_name)
        return process_hao_case(
            input_path, os.path.join(output_path, input_folder_name),
            ct_path, class_map, subfolder_name=subfolder_name)

    # ---- NEW: copy input patient folder to output first ----
    if output_path is not None:
        dst_path = os.path.join(output_path, input_folder_name)
        if not os.path.exists(dst_path):
            shutil.copytree(input_path, dst_path)
        else:
            pass
    # --------------------------------------------------------
    
    seg_path = os.path.join(input_path, reference_file_name)
    if not os.path.exists(seg_path):
        # e.g. vertebrae-only cases have no liver mask: use any mask of the case as reference
        seg_dir = os.path.join(input_path, subfolder_name)
        seg_path = os.path.join(seg_dir, sorted(f for f in os.listdir(seg_dir) if f.endswith('.nii.gz'))[0])
    img = nib.load(seg_path)
    
    segmentation_dict = read_all_segmentations(
        folder_path=input_path,
        organ_list=organ_list,
        subfolder_name=subfolder_name,
        target_axcodes=nib.aff2axcodes(img.affine) 
    )

    segmentation = combine_segmentation_dict(segmentation_dict, class_map)
    patient_id = os.path.basename(input_path)

    # locate the case CT for the shapekit_pro vertebrae engine (optional)
    ct_path = os.path.join(input_path, ct_file_name)
    if not os.path.exists(ct_path) and ct_root is not None:
        ct_path = os.path.join(ct_root, input_folder_name, ct_file_name)

    postprocessed_segmentation_dict = process_organs(
        segmentation_dict, 
        img,
        segmentation,
        target_organs,
        patient_id = patient_id,
        logger = logging,
        ct_path = ct_path,
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
        return sub_folder
    except Exception as e:
        logging.error(f"[CRASH] {sub_folder}: {e}")
        traceback.print_exc()
        return sub_folder


def run_in_parallel(sub_folders, input_folder, output_folder, max_workers=4, tqdm_ncols=80):
    logging.info(
        f"\n\n[INFO] Start processing {len(sub_folders)} cases with up to {max_workers} workers.\n\n"
    )

    args_list = [(sub_folder, input_folder, output_folder) for sub_folder in sub_folders]

    failed = []
    with Pool(processes=max_workers) as pool:
        for failure in tqdm(
            pool.imap_unordered(process_case_wrapper, args_list),
            total=len(args_list),
            desc="Processing cases",
            unit="case",
            ncols=tqdm_ncols,
        ):
            if failure is not None:
                failed.append(failure)
    return failed
############################## Parallel Execution with multiprocessing.Pool ##############################




parser = argparse.ArgumentParser(description="Anatomical-aware post-processing")
parser.add_argument('--input_folder', type=str, help='Input files folder location, /path/to/input/data')
parser.add_argument('--output_folder', type=str, help='Output files folder location, /path/to/save/results')
parser.add_argument('--log_folder', type=str, default='./logs/task_001', help='Logging folder location')
parser.add_argument('--csv', type=str, default=None, help='Guidence csv file telling ShapeKit specific ones for processing')
parser.add_argument('--cpu_count', type=int, default=cpu_count(), help='Number of CPU cores to use for parallel processing (default: system max)')
parser.add_argument('--continue_prediction', action="store_true", help='If continue from last processing record')
parser.add_argument('--tqdm_ncols', type=int, default=100, help='Width of tqdm progress bar in characters')
parser.add_argument('--vertebrae_engine', choices=['shapekit', 'shapekit_songlin', 'shapekit_pro', 'shapekit_hao', 'shapekit_davir'],
                    help='Override config.yaml vertebrae engine for this run')
parser.add_argument('--vertebrae_only', action='store_true', help='Process vertebrae only')
parser.add_argument('--ct_root', help='Override external case CT root for this run')
parser.add_argument('--case', action='append', help='Process only this case folder (repeatable)')
args = parser.parse_args()
if args.vertebrae_engine is not None:
    vertebrae_engine = args.vertebrae_engine
if args.vertebrae_only:
    target_organs = {'vertebrae'}
if args.ct_root is not None:
    ct_root = args.ct_root



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

    if input_folder is None or output_folder is None:
        parser.error('--input_folder and --output_folder are required')
    if args.cpu_count < 1:
        parser.error('--cpu_count must be at least 1')
    if vertebrae_engine == 'shapekit_hao':
        if target_organs != {'vertebrae'}:
            parser.error('shapekit_hao requires --vertebrae_only')
        if args.continue_prediction:
            parser.error('shapekit_hao uses new case directories; do not use --continue_prediction')
        from pathlib import Path
        hao_output = Path(output_folder).resolve()
        for source in (input_folder, ct_root):
            if source is not None:
                hao_source = Path(source).resolve()
                if (hao_source == hao_output or hao_source in hao_output.parents
                        or hao_output in hao_source.parents):
                    parser.error('Output root must be disjoint from input and CT roots')

    sub_folders = [sf for sf in os.listdir(input_folder)
                   if not sf.startswith('.') and os.path.isdir(os.path.join(input_folder, sf))]
    sub_folders.sort()
    sub_folders = set(sub_folders)
    if args.case:
        requested = set(args.case)
        if not requested <= sub_folders:
            parser.error('Requested case directory not found')
        sub_folders &= requested

    
    if args.csv is not None:
        csv_cases = read_cases_from_csv(args.csv)
        before = len(sub_folders)
        sub_folders = sub_folders & csv_cases
        after = len(sub_folders)
        print(f"[INFO] CSV filtering enabled: {after}/{before} cases kept")


    if args.continue_prediction:
        continue_cases = set(
            check_unprocessed_cases(
                input_folder=input_folder,
                output_folder=output_folder,
                csv_path="continue.csv",
            )
        )
        before = len(sub_folders)
        sub_folders = sub_folders & continue_cases
        after = len(sub_folders)
        print(
            f"[INFO] Resume enabled: {after}/{before} cases remain after continue.csv"
        )

    if not sub_folders:
        parser.error('No cases selected')
    max_workers = min(args.cpu_count, len(sub_folders), max(1, multiprocessing.cpu_count() - 1))
    print(f"[INFO] Starting... with {max_workers} multiprocess ...")
    print(f"[INFO] Input files dir: {input_folder}")
    print(f"[INFO] Output files dir: {output_folder}")
    print(f"[INFO] Logging dir: {args.log_folder}\n\n")
    failed = run_in_parallel(
        sorted(sub_folders),
        input_folder,
        output_folder,
        max_workers=max_workers,
        tqdm_ncols=args.tqdm_ncols,
    )
    if failed:
        print(f'[ERROR] Failed cases: {", ".join(sorted(failed))}')
        raise SystemExit(1)
