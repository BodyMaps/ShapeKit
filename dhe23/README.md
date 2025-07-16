Post-Processing Pipeline for CT Segmentation
==============================================

This script performs post-processing on CT segmentation datasets.
It allows remapping class IDs using customizable class maps and supports
multiprocessing to accelerate processing.

------------------------------
Directory Structure
------------------------------

All input and output data should be stored under the ./data directory:

./data/
├── <source_dir_1>/         # Raw labeled segmentation data
├── <source_dir_2>/         # Another dataset
├── ...

- <source_dir>: Input directory with segmentation label files.
- <target_dir>: Output directory for processed results.

You must specify both --source_dir and --target_dir when running the script.

------------------------------
Class Map
------------------------------

To remap segmentation class labels, define your mappings in the 'class_map.py' file.

Example (inside class_map.py):

    class_map_example = {
        "1": "liver",
        "2": "kidney",
        "3": "spleen",
        ...
    }

You can define multiple mappings and select one with the --class_map argument.

------------------------------
Multiprocessing
------------------------------

Use the --n_jobs argument to enable multiprocessing.

Example:
    --n_jobs 5

This uses 5 parallel processes. You can adjust this depending on your CPU.

------------------------------
Example Commands
------------------------------

Use the yaml file to post-process:

    python -W ignore post_processing.py --config config.yaml

or run the following commands to post-process different datasets:

    python -W ignore post_processing.py --source_dir data/PanTS --target_dir data/PanTS_processed --n_jobs 5

    python -W ignore post_processing.py --source_dir data/PanTSV2 --target_dir data/PanTSV2_processed --n_jobs 5

    python -W ignore post_processing.py --source_dir data/JuMaMini_noCT --target_dir data/JuMaMini_noCT_processed --n_jobs 5

    python -W ignore post_processing.py --source_dir /mnt/bodymaps/mask_only/JuMaMini/JuMaMini --target_dir /mnt/T9/temp_data_to_delete_very_soon/zzhou82/JuMaMini_dhe23 --n_jobs 80

    python -W ignore post_processing.py --source_dir /mnt/bodymaps/mask_only/JuMa/JuMa --target_dir /mnt/T9/temp_data_to_delete_very_soon/zzhou82/JuMa_dhe23 --n_jobs 64

    python -W ignore post_processing.py --source_dir /mnt/bodymaps/mask_only/AbdomenAtlas1.1/AbdomenAtlas1.1 --target_dir /mnt/T9/temp_data_to_delete_very_soon/zzhou82/AA1.1_dhe23 --n_jobs 70 --start_idx 1000 --end_idx -1

    python -W ignore post_processing.py --source_dir ../data/AbdomenAtlasPro --target_dir outputs --n_jobs 64

------------------------------
Requirements
------------------------------

- Python 3.9 or higher

To install dependencies (if requirements.txt is available):

    pip install -r requirements.txt

------------------------------
Contact
------------------------------

For questions or help, please open an issue or contact the maintainer.
