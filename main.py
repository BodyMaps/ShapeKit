import argparse
import json
import multiprocessing
import os
from pathlib import Path
from multiprocessing import cpu_count
import uuid
from utils.organs_postprocessing import *
from utils.vertebrae_postprocessing import postprocessing_vertebrae
from utils.vertebrae_iterative import postprocessing_vertebrae as postprocessing_vertebrae_songlin
from utils.vertebrae_pro import postprocessing_vertebrae_pro
import logging
import yaml
import traceback

from multiprocessing import Pool
from tqdm import tqdm
import csv

import shutil



"""
mail: jliu452@uw.edu
last systematic update: Dec 23, 2025

"""
##############################################################
PROJECT_ROOT = Path(__file__).resolve().parent
with (PROJECT_ROOT / 'config.yaml').open('r') as f:
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
COMPLETION_MARKER = '.shapekit_complete'

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
        processed = os.path.isfile(os.path.join(out_dir, COMPLETION_MARKER))

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
            case_id = row[key].strip()
            if case_id:
                cases.append(case_id)
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
        has_vertebrae = any(
            name.startswith('vertebrae_') and mask is not None and np.any(mask)
            for name, mask in segmentation_dict.items()
        )
        if not has_vertebrae:
            logger.info(f"[INFO] {patient_id}, No vertebra masks found; skipping vertebra processing.")
        elif vertebrae_engine == 'shapekit_pro':
            segmentation_dict = postprocessing_vertebrae_pro(
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



def _copy_case_metadata(input_path: Path, staging_path: Path):
    """Copy non-segmentation case files while outputs are built from scratch."""
    staging_path.mkdir(parents=True)
    for source in input_path.iterdir():
        if source.name in {subfolder_name, 'combined_labels.nii.gz', COMPLETION_MARKER}:
            continue
        destination = staging_path / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _commit_case_output(staging_path: Path, final_path: Path):
    """Replace a case directory only after its staged output is complete."""
    backup_path = None
    if final_path.exists():
        backup_path = final_path.with_name(
            f'.{final_path.name}.backup-{uuid.uuid4().hex}'
        )
        os.replace(final_path, backup_path)

    try:
        os.replace(staging_path, final_path)
    except Exception:
        if backup_path is not None and backup_path.exists() and not final_path.exists():
            os.replace(backup_path, final_path)
        raise
    else:
        if backup_path is not None:
            try:
                shutil.rmtree(backup_path)
            except OSError as error:
                logging.warning('Could not remove backup %s: %s', backup_path, error)


def main(input_path, input_folder_name, output_path=None):
    """
    input_path: the folder path
    """
    
    if output_path is None:
        raise ValueError('output_path is required')

    input_path = Path(input_path).resolve()
    output_root = Path(output_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_path = output_root / input_folder_name
    staging_path = output_root / f'.{input_folder_name}.tmp-{uuid.uuid4().hex}'

    try:
        seg_path = input_path / reference_file_name
        img = nib.load(str(seg_path))
        if len(img.shape) != 3:
            raise ValueError(f'Reference image must be 3D, got shape {img.shape}')

        segmentation_dict = read_all_segmentations(
            folder_path=str(input_path),
            organ_list=organ_list,
            subfolder_name=subfolder_name,
            reference_img=img,
        )
        segmentation = combine_segmentation_dict(segmentation_dict, class_map)
        patient_id = input_path.name

        ct_path = input_path / ct_file_name
        if not ct_path.exists() and ct_root is not None:
            ct_path = Path(ct_root) / input_folder_name / ct_file_name

        postprocessed_segmentation_dict = process_organs(
            segmentation_dict,
            img,
            segmentation,
            target_organs,
            patient_id=patient_id,
            logger=logging,
            ct_path=str(ct_path),
        )

        _copy_case_metadata(input_path, staging_path)
        save_and_combine_segmentations(
            processed_segmentation_dict=postprocessed_segmentation_dict,
            class_map=class_map,
            reference_img=img,
            output_folder=str(staging_path),
            if_save_combined=save_combined_label_bool,
        )
        marker = {
            'case_id': input_folder_name,
            'reference_file': reference_file_name,
            'saved_combined_label': save_combined_label_bool,
        }
        with (staging_path / COMPLETION_MARKER).open('w') as marker_file:
            json.dump(marker, marker_file, sort_keys=True)
            marker_file.write('\n')

        _commit_case_output(staging_path, final_path)
    finally:
        if staging_path.exists():
            try:
                shutil.rmtree(staging_path)
            except OSError as error:
                logging.warning('Could not remove staging directory %s: %s', staging_path, error)



############################## Parallel Execution with multiprocessing.Pool ##############################
post_logger = logging.getLogger("postprocessing")

def process_case_wrapper(args):
    sub_folder, input_folder, output_folder = args
    try:
        input_path = os.path.join(input_folder, sub_folder)
        main(input_path, sub_folder, output_folder)
        post_logger.info(f"[ShapeKit] Successfully processed {sub_folder}")
        return sub_folder, None
    except Exception as error:
        error_details = traceback.format_exc()
        logging.error(f"[CRASH] {sub_folder}: {error}\n{error_details}")
        return sub_folder, f"{type(error).__name__}: {error}"


def run_in_parallel(sub_folders, input_folder, output_folder, max_workers=4, tqdm_ncols=80):
    logging.info(
        f"\n\n[INFO] Start processing {len(sub_folders)} cases with up to {max_workers} workers.\n\n"
    )

    args_list = [(sub_folder, input_folder, output_folder) for sub_folder in sub_folders]

    with Pool(processes=max_workers) as pool:
        failures = []
        for case_id, error in tqdm(
            pool.imap_unordered(process_case_wrapper, args_list),
            total=len(args_list),
            desc="Processing cases",
            unit="case",
            ncols=tqdm_ncols,
        ):
            if error is not None:
                failures.append((case_id, error))

    if failures:
        summary = '; '.join(f'{case_id} ({error})' for case_id, error in failures)
        raise RuntimeError(f'{len(failures)} case(s) failed: {summary}')
############################## Parallel Execution with multiprocessing.Pool ##############################


def _available_memory_bytes():
    """Best-effort available-memory lookup for Linux hosts."""
    try:
        with open('/proc/meminfo') as meminfo:
            for line in meminfo:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def resolve_worker_count(requested, case_count, memory_per_worker_gb=4.0):
    if requested < 1:
        raise ValueError('--cpu_count must be at least 1')
    if memory_per_worker_gb <= 0:
        raise ValueError('--memory_per_worker_gb must be greater than 0')
    if case_count < 1:
        return 0

    cpu_limit = max(1, multiprocessing.cpu_count() - 1)
    worker_count = min(requested, case_count, cpu_limit)
    available = _available_memory_bytes()
    if available is not None:
        bytes_per_worker = memory_per_worker_gb * 1024 ** 3
        memory_limit = max(1, int((available * 0.75) // bytes_per_worker))
        worker_count = min(worker_count, memory_limit)
    return max(1, worker_count)


def build_parser():
    parser = argparse.ArgumentParser(description="Anatomical-aware post-processing")
    parser.add_argument('--input_folder', required=True, help='Input files folder location')
    parser.add_argument('--output_folder', required=True, help='Output files folder location')
    parser.add_argument('--log_folder', default='./logs/task_001', help='Logging folder location')
    parser.add_argument('--csv', default=None, help='CSV file selecting cases for processing')
    parser.add_argument(
        '--cpu_count',
        type=int,
        default=min(4, max(1, cpu_count() - 1)),
        help='Maximum worker processes (default: up to 4)',
    )
    parser.add_argument(
        '--memory_per_worker_gb',
        type=float,
        default=4.0,
        help='Estimated memory required per worker, used to cap parallelism',
    )
    parser.add_argument('--continue_prediction', action="store_true", help='Resume incomplete cases')
    parser.add_argument('--tqdm_ncols', type=int, default=100, help='Width of tqdm progress bar')
    return parser


def configure_logging(log_folder):
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_folder, 'debug.log'),
        level=logging.DEBUG,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
    )
    post_logger.setLevel(logging.INFO)
    post_logger.propagate = False
    post_logger.handlers.clear()
    post_handler = logging.FileHandler(os.path.join(log_folder, 'postprocessing.log'))
    post_handler.setFormatter(logging.Formatter(
        '[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    post_logger.addHandler(post_handler)


def cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    input_folder = Path(args.input_folder).resolve()
    output_folder = Path(args.output_folder).resolve()

    if not input_folder.is_dir():
        parser.error(f'input folder does not exist: {input_folder}')
    if output_folder == input_folder or input_folder in output_folder.parents:
        parser.error('output folder must not be the input folder or inside it')
    if args.cpu_count < 1:
        parser.error('--cpu_count must be at least 1')
    if args.memory_per_worker_gb <= 0:
        parser.error('--memory_per_worker_gb must be greater than 0')

    configure_logging(args.log_folder)
    sub_folders = sorted(
        path.name for path in input_folder.iterdir() if path.is_dir()
    )

    if args.csv is not None:
        csv_cases = read_cases_from_csv(args.csv)
        before = len(sub_folders)
        sub_folders = [case for case in sub_folders if case in csv_cases]
        print(f"[INFO] CSV filtering enabled: {len(sub_folders)}/{before} cases kept")

    if args.continue_prediction:
        continue_cases = set(check_unprocessed_cases(
            input_folder=str(input_folder),
            output_folder=str(output_folder),
            csv_path=str(output_folder / 'continue.csv'),
        ))
        before = len(sub_folders)
        sub_folders = [case for case in sub_folders if case in continue_cases]
        print(f"[INFO] Resume enabled: {len(sub_folders)}/{before} cases remain")

    if not sub_folders:
        print('[INFO] No cases to process.')
        return 0

    max_workers = resolve_worker_count(
        args.cpu_count,
        len(sub_folders),
        args.memory_per_worker_gb,
    )
    print(f"[INFO] Starting with {max_workers} worker(s)")
    print(f"[INFO] Input files dir: {input_folder}")
    print(f"[INFO] Output files dir: {output_folder}")
    print(f"[INFO] Logging dir: {args.log_folder}\n")
    run_in_parallel(
        sub_folders,
        str(input_folder),
        str(output_folder),
        max_workers=max_workers,
        tqdm_ncols=args.tqdm_ncols,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
