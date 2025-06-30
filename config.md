<h1 align="center">Config Setting</h1>



1. `subfolder_name`: the name of subfolder in the input/output folder. For example, the input folder structure:
    ```
    INPUT or OUTPUT
    └── case_001
        └── segmentations <- (subfolder_name)
                ├── liver.nii.gz
                ...
                └── veins.nii.gz
    ```

2. `class_map`: the mapping dict of organ and their labels.

3. `target_organs`: the organs selected for postprocessing. 

4. `organ_adjacency_map`: a dictionary used in the `reassign_false_positives` function. It defines pairs of anatomically adjacent organs where false positive segmentations are likely to occur between each other.

5. `affine_reference_file_name`: file to load affine reference info.

6. `if_save_combined_label`: booling parameter, whether to save the combined labels as a *.nii.gz* file after processing. For example:

    ```
    OUTPUT
    └── case_001
        ├── combined_labels.nii.gz <- (if_save_combined_label)
        └── segmentations
                ├── liver.nii.gz
                ...
                └── veins.nii.gz
    ```