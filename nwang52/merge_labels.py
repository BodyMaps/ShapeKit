import os
import argparse
import nibabel as nib
import numpy as np
import logging
import json
import sys

# Import class maps from separate module
from class_maps import available_class_maps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_segmentations(input_dir, class_map, case_folders=None):
    """
    Merge individual organ segmentation files into a single multi-label image
    and save directly in each case folder.
    
    Args:
        input_dir: Directory containing case folders with segmentation files
        class_map: Dictionary mapping class IDs to organ names
        case_folders: List of specific case folders to process, if None process all
    """
    # Get list of case folders to process
    if case_folders is None:
        case_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    # Create inverted map for lookup (organ_name -> class_id)
    organ_to_label = {v: k for k, v in class_map.items()}
    
    # Log available organs in class map
    logging.info(f"Using class map with {len(class_map)} classes")
    for class_id, organ_name in class_map.items():
        logging.info(f"  {class_id}: {organ_name}")
    
    total_cases = len(case_folders)
    success_count = 0
    
    # Process each case
    for i, case_folder in enumerate(case_folders, 1):
        case_path = os.path.join(input_dir, case_folder)
        seg_dir = os.path.join(case_path, "segmentations")
        
        # Check if segmentations directory exists
        if not os.path.exists(seg_dir):
            logging.warning(f"Segmentations directory not found for case {case_folder}, skipping")
            continue
        
        logging.info(f"Processing case {i}/{total_cases}: {case_folder}")
        
        # Get all segmentation files
        seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.nii.gz')]
        
        if not seg_files:
            logging.warning(f"No segmentation files found in {seg_dir}, skipping")
            continue
        
        # Load first segmentation to get reference image properties
        first_seg_path = os.path.join(seg_dir, seg_files[0])
        reference_nii = nib.load(first_seg_path)
        reference_shape = reference_nii.shape
        reference_affine = reference_nii.affine
        reference_dtype = reference_nii.get_data_dtype()
        
        # Create empty combined labels image
        combined_labels = np.zeros(reference_shape, dtype=reference_dtype)
        
        # Track which organs were found
        found_organs = []
        
        # Process each organ segmentation file
        for seg_file in seg_files:
            # Extract organ name from filename
            organ_name = seg_file.split('.')[0]  # Remove file extension
            
            # Check if organ is in class map
            if organ_name not in organ_to_label:
                logging.warning(f"Organ '{organ_name}' not found in class map, skipping")
                continue
            
            # Get label ID for this organ
            label_id = organ_to_label[organ_name]
            
            # Load segmentation
            seg_path = os.path.join(seg_dir, seg_file)
            seg_nii = nib.load(seg_path)
            seg_data = seg_nii.get_fdata() > 0  # Binarize
            
            # Check if segmentation is empty
            if not np.any(seg_data):
                logging.info(f"  {organ_name} segmentation is empty")
                continue
            
            # Add to combined labels
            combined_labels[seg_data] = label_id
            found_organs.append(organ_name)
            
            # Log information
            logging.info(f"  Added {organ_name} (label {label_id}) with volume {np.sum(seg_data)}")
        
        # Save combined labels directly in the case directory
        output_file = os.path.join(case_path, "combined_labels.nii.gz")
        combined_nii = nib.Nifti1Image(combined_labels, reference_affine)
        nib.save(combined_nii, output_file)
        
        # Create a label mapping file for reference
        mapping_file = os.path.join(case_path, "label_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump({str(k): v for k, v in class_map.items()}, f, indent=2)
        
        # Log summary
        logging.info(f"Created combined labels file with {len(found_organs)} organs: {', '.join(found_organs)}")
        logging.info(f"Saved to {output_file}")
        success_count += 1
    
    logging.info(f"Processing complete. Processed {success_count}/{total_cases} cases successfully.")
    return success_count

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Merge organ segmentations into a single multi-label image")
    parser.add_argument("--input_dir", "-i", type=str, required=True, 
                        help="Input directory containing case folders with segmentations")
    parser.add_argument("--class_map", "-c", type=str, required=True,
                        choices=list(available_class_maps.keys()),
                        help="Class mapping to use")
    parser.add_argument("--list_maps", "-l", action="store_true",
                        help="List available class maps and exit")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # List available class maps if requested
    if args.list_maps:
        print("Available class maps:")
        for map_name in available_class_maps.keys():
            print(f"  - {map_name}")
        sys.exit(0)
    
    # Get the selected class map
    class_map = available_class_maps[args.class_map]
    
    # Run merge process
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Using class map: {args.class_map}")
    
    merge_segmentations(args.input_dir, class_map)
