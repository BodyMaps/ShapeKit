import os
import shutil
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cleanup_incomplete_cases(base_dir, dry_run=False):
    """
    Check BDMAP folders for combined_labels.nii.gz and remove incomplete ones
    
    Args:
        base_dir: Path to the directory containing BDMAP folders
        dry_run: If True, only show what would be deleted without actually deleting
    
    Returns:
        Tuple of (total_folders, incomplete_folders, deleted_folders)
    """
    if not os.path.exists(base_dir):
        logging.error(f"Base directory does not exist: {base_dir}")
        return 0, 0, 0
    
    # Get all BDMAP folders
    all_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    bdmap_folders = [f for f in all_folders if f.startswith('BDMAP_')]
    
    if not bdmap_folders:
        logging.warning(f"No BDMAP folders found in {base_dir}")
        return 0, 0, 0
    
    logging.info(f"Found {len(bdmap_folders)} BDMAP folders to check")
    
    incomplete_folders = []
    deleted_folders = []
    
    # Check each BDMAP folder
    for folder_name in sorted(bdmap_folders):
        folder_path = os.path.join(base_dir, folder_name)
        combined_labels_path = os.path.join(folder_path, "combined_labels.nii.gz")
        
        if not os.path.exists(combined_labels_path):
            incomplete_folders.append(folder_name)
            logging.info(f"Missing combined_labels.nii.gz: {folder_name}")
            
            if not dry_run:
                try:
                    shutil.rmtree(folder_path)
                    deleted_folders.append(folder_name)
                    logging.info(f"Deleted folder: {folder_name}")
                except Exception as e:
                    logging.error(f"Failed to delete {folder_name}: {str(e)}")
            else:
                logging.info(f"[DRY RUN] Would delete: {folder_name}")
        else:
            logging.debug(f"Complete: {folder_name}")
    
    # Summary
    total_folders = len(bdmap_folders)
    incomplete_count = len(incomplete_folders)
    deleted_count = len(deleted_folders)
    
    logging.info(f"\nSummary:")
    logging.info(f"  Total BDMAP folders: {total_folders}")
    logging.info(f"  Incomplete folders: {incomplete_count}")
    
    if dry_run:
        logging.info(f"  Would be deleted: {incomplete_count}")
    else:
        logging.info(f"  Successfully deleted: {deleted_count}")
        if deleted_count != incomplete_count:
            logging.warning(f"  Failed to delete: {incomplete_count - deleted_count}")
    
    return total_folders, incomplete_count, deleted_count

def list_incomplete_cases(base_dir):
    """
    List all incomplete BDMAP cases without deleting them
    
    Args:
        base_dir: Path to the directory containing BDMAP folders
    """
    if not os.path.exists(base_dir):
        logging.error(f"Base directory does not exist: {base_dir}")
        return
    
    # Get all BDMAP folders
    all_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    bdmap_folders = [f for f in all_folders if f.startswith('BDMAP_')]
    
    if not bdmap_folders:
        logging.warning(f"No BDMAP folders found in {base_dir}")
        return
    
    logging.info(f"Checking {len(bdmap_folders)} BDMAP folders...")
    
    incomplete_folders = []
    complete_folders = []
    
    # Check each BDMAP folder
    for folder_name in sorted(bdmap_folders):
        folder_path = os.path.join(base_dir, folder_name)
        combined_labels_path = os.path.join(folder_path, "combined_labels.nii.gz")
        
        if not os.path.exists(combined_labels_path):
            incomplete_folders.append(folder_name)
        else:
            complete_folders.append(folder_name)
    
    # Display results
    logging.info(f"\nResults:")
    logging.info(f"  Total BDMAP folders: {len(bdmap_folders)}")
    logging.info(f"  Complete folders: {len(complete_folders)}")
    logging.info(f"  Incomplete folders: {len(incomplete_folders)}")
    
    if incomplete_folders:
        logging.info(f"\nIncomplete folders (missing combined_labels.nii.gz):")
        for folder in incomplete_folders[:20]:  # Show first 20
            logging.info(f"  - {folder}")
        if len(incomplete_folders) > 20:
            logging.info(f"  ... and {len(incomplete_folders) - 20} more")
    
    if complete_folders:
        logging.info(f"\nComplete folders (first 10):")
        for folder in complete_folders[:10]:
            logging.info(f"  - {folder}")
        if len(complete_folders) > 10:
            logging.info(f"  ... and {len(complete_folders) - 10} more")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cleanup incomplete BDMAP cases")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input directory containing BDMAP folders")
    parser.add_argument("--dry-run", "-d", action="store_true",
                        help="Show what would be deleted without actually deleting")
    parser.add_argument("--list-only", "-l", action="store_true",
                        help="Only list incomplete cases without deletion")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    base_dir = args.input
    
    logging.info(f"Target directory: {base_dir}")
    
    if args.list_only:
        logging.info("List mode: Only showing incomplete cases")
        list_incomplete_cases(base_dir)
    else:
        if args.dry_run:
            logging.info("Dry run mode: No files will be deleted")
        else:
            logging.warning("DELETION mode: Incomplete folders will be permanently deleted!")
            response = input("Are you sure you want to continue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                logging.info("Operation cancelled by user")
                exit(0)
        
        total, incomplete, deleted = cleanup_incomplete_cases(base_dir, dry_run=args.dry_run)
        
        if not args.dry_run and deleted > 0:
            logging.info(f"\nCleanup completed. {deleted} folders were deleted.")
