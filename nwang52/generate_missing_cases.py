import os
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_missing_cases_list(input_dir, output_file, start_case=1, end_case=1000):
    """
    Generate a list of missing BDMAP cases in the specified range
    
    Args:
        input_dir: Directory to scan for existing BDMAP cases
        output_file: Path to output txt file containing missing case names
        start_case: Starting case number (default: 1)
        end_case: Ending case number (default: 1000)
    """
    if not os.path.exists(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return False
    
    # Get all existing case folders
    case_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    # Filter and extract BDMAP case numbers
    existing_cases = set()
    for folder in case_folders:
        if folder.startswith('BDMAP_'):
            try:
                # Extract the number part after 'BDMAP_'
                number_part = folder.split('_')[1]
                case_number = int(number_part)
                existing_cases.add(case_number)
                logging.debug(f"Found existing BDMAP case: {folder} -> {case_number}")
            except (ValueError, IndexError):
                logging.warning(f"Invalid BDMAP case format: {folder}")
                continue
    
    logging.info(f"Found {len(existing_cases)} existing BDMAP cases in {input_dir}")
    
    # Generate complete range of expected cases
    expected_cases = set(range(start_case, end_case + 1))
    logging.info(f"Expected case range: BDMAP_{start_case:08d} to BDMAP_{end_case:08d} ({len(expected_cases)} total)")
    
    # Find missing cases
    missing_cases = expected_cases - existing_cases
    missing_case_names = [f"BDMAP_{case_num:08d}" for case_num in sorted(missing_cases)]
    
    logging.info(f"Found {len(missing_cases)} missing cases")
    
    if missing_cases:
        # Show some examples of missing cases
        sample_missing = sorted(list(missing_cases))[:10]
        sample_names = [f"BDMAP_{num:08d}" for num in sample_missing]
        logging.info(f"Sample missing cases (first 10): {sample_names}")
        
        # Save missing cases to txt file
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for case_name in missing_case_names:
                    f.write(case_name + '\n')
            
            logging.info(f"Missing cases list saved to: {output_file}")
            logging.info(f"Total missing cases written: {len(missing_case_names)}")
            return True
            
        except Exception as e:
            logging.error(f"Error writing to output file {output_file}: {str(e)}")
            return False
    else:
        logging.info("No missing cases found in the specified range")
        # Create empty file to indicate no missing cases
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# No missing cases found\n")
            logging.info(f"Empty missing cases list saved to: {output_file}")
            return True
        except Exception as e:
            logging.error(f"Error writing to output file {output_file}: {str(e)}")
            return False

def generate_existing_cases_list(input_dir, output_file):
    """
    Generate a list of existing BDMAP cases
    
    Args:
        input_dir: Directory to scan for existing BDMAP cases
        output_file: Path to output txt file containing existing case names
    """
    if not os.path.exists(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return False
    
    # Get all existing case folders
    case_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    # Filter and collect BDMAP cases
    existing_cases = []
    for folder in case_folders:
        if folder.startswith('BDMAP_'):
            try:
                # Extract the number part after 'BDMAP_'
                number_part = folder.split('_')[1]
                case_number = int(number_part)
                existing_cases.append((case_number, folder))
                logging.debug(f"Found existing BDMAP case: {folder} -> {case_number}")
            except (ValueError, IndexError):
                logging.warning(f"Invalid BDMAP case format: {folder}")
                continue
    
    # Sort by case number
    existing_cases.sort(key=lambda x: x[0])
    existing_case_names = [folder for _, folder in existing_cases]
    
    logging.info(f"Found {len(existing_cases)} existing BDMAP cases in {input_dir}")
    
    if existing_cases:
        # Show some examples
        sample_cases = existing_case_names[:10]
        logging.info(f"Sample existing cases (first 10): {sample_cases}")
        
        # Save existing cases to txt file
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for case_name in existing_case_names:
                    f.write(case_name + '\n')
            
            logging.info(f"Existing cases list saved to: {output_file}")
            logging.info(f"Total existing cases written: {len(existing_case_names)}")
            return True
            
        except Exception as e:
            logging.error(f"Error writing to output file {output_file}: {str(e)}")
            return False
    else:
        logging.warning("No BDMAP cases found in the input directory")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate missing or existing BDMAP cases list")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input directory to scan for existing BDMAP cases")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output txt file path to save the cases list")
    parser.add_argument("--mode", "-m", type=str, choices=['missing', 'existing'], default='missing',
                        help="Generate missing cases list or existing cases list (default: missing)")
    parser.add_argument("--start", "-s", type=int, default=1,
                        help="Starting case number for missing cases mode (default: 1)")
    parser.add_argument("--end", "-e", type=int, default=1000,
                        help="Ending case number for missing cases mode (default: 1000)")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug mode to show detailed information")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set debug logging level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    input_dir = args.input
    output_file = args.output
    mode = args.mode
    start_case = args.start
    end_case = args.end
    
    logging.info(f"Mode: {mode}")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output file: {output_file}")
    
    if mode == 'missing':
        logging.info(f"Case range: {start_case} to {end_case}")
        success = generate_missing_cases_list(input_dir, output_file, start_case, end_case)
    else:  # mode == 'existing'
        success = generate_existing_cases_list(input_dir, output_file)
    
    if success:
        logging.info("Cases list generation completed successfully")
    else:
        logging.error("Cases list generation failed")
        exit(1)
