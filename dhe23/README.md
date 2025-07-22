Post-Processing Pipeline for CT Segmentation
==============================================

This script performs post-processing on CT segmentation datasets.
It allows remapping class IDs using customizable class maps and supports
multiprocessing to accelerate processing.

------------------------------
Directory Structure
------------------------------

All input and output data should be specified with input_folder and output_folder, either using configuration file or arguments:
```
INPUT or OUTPUT
└── CASE_001
    ├── combined_labels.nii.gz (optional)
    └── segmentations
            ├── liver.nii.gz
            ...
            └── pancreas_head.nii.gz
```

------------------------------
Supported Class Map
------------------------------

A list of supported organs is available in class_maps.py. Please refer to that file for details. Please make sure the organ name of your segmentation file is exactly the same with the supported organ name.

------------------------------
Example Commands
------------------------------

Use the yaml file to post-process:

    python main.py --config config.yaml

or run the following commands to post-process different datasets:

    python main.py --input_folder data --output_folder outputs --cpu_count 4

    # For Zongwei's use. To be removed.
    python -W ignore main.py --input_folder /mnt/bodymaps/mask_only/JuMaMini/JuMaMini --output_folder /mnt/T9/temp_data_to_delete_very_soon/zzhou82/JuMaMini_dhe23 --cpu_count 80

    python -W ignore main.py --input_folder /mnt/bodymaps/mask_only/JuMa/JuMa --output_folder /mnt/T9/temp_data_to_delete_very_soon/zzhou82/JuMa_dhe23 --cpu_count 64

    python -W ignore main.py --input_folder /mnt/bodymaps/mask_only/AbdomenAtlas1.1/AbdomenAtlas1.1 --output_folder /mnt/T9/temp_data_to_delete_very_soon/zzhou82/AA1.1_dhe23 --cpu_count 70 --start_idx 1000 --end_idx -1

    # for Dongli's use. To be removed.
    python -W ignore main.py --input_folder data/PanTS --output_folder data/PanTS_processed --cpu_count 5

    python -W ignore main.py --input_folder data/PanTSV2 --output_folder data/PanTSV2_processed --cpu_count 5

    python -W ignore main.py --input_folder data/JuMaMini_noCT --output_folder data/JuMaMini_noCT_processed --cpu_count 5
    
    python -W ignore main.py --input_folder data/AbdomenAtlasPro --output_folder outputs --cpu_count 64

    python main.py --input_folder data --output_folder outputs --cpu_count 1

------------------------------
Logging
------------------------------

By default, ``verbose`` is set to ``False``, and only the progress bar is shown. Since the order of logging messages in multiprocessing is not guaranteed, this helps reduce messy and interleaved outputs from concurrent processes.

If you want to see detailed logging for each organ being processed, you can set verbose to True.

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
