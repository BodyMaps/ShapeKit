<h1 align="center">Config Setting</h1>



1. `subfolder_name`: the name of subfolder in the input/output folder containing organs `.nii.gz` files. For example:
    ```
    INPUT / OUTPUT (--input_folder / --output_folder)
    └── case_001
        └── segmentations <- (subfolder_name)
                ├── liver.nii.gz
                ...
                └── veins.nii.gz
    ```

2. `class_map`: the label mapping dict of organ and their labels.
    > [!WARNING]
    > This parameter will be deprecated soon.

    All organs on this list will be read and loaded, but only the ones listed in target_organs will be processed by ShapeKit.

3. `target_organs`: the organs selected for postprocessing. 

    By adding or deleting the organs listed, you can choose which organs you want to process. For example:
    ```
        target_organs:
            - bladder
            - colon
            - duodenum
            - femur
            - intestine
            - kidney
            - liver
            - lung
            - pancreas
    ```


4. `organ_adjacency_map`: a dictionary used in the `reassign_false_positives` function. 
    
   This section identifies organs that sit close together where the AI might mislabel a border. By listing these anatomical neighbors, you help the software distinguish between touching structures—like the liver and pancreas—to ensure your results are accurate.

    Exmaple:
    ```
    organ_adjacency_map:
        lung_left: [postcava]
        lung_right: [postcava]
        liver: [kidney_right, pancreas]
    ```

    This means that during segmentation:
	(1) Parts of the predicted `lung_left` may be false positives that actually belong to `postcava`.
    (2) Similarly, `liver` may mistakenly include areas from `kidney_right` or `pancreas`.

    **Note: This map is one-directional**, i.e., if `lung_left` → `postcava` is defined, it does not imply the reverse (`postcava` → `lung_left`). This directionality reflects common misclassification patterns, not anatomical symmetry.

5. `affine_reference_file_name`: file to load affine reference info.

6. `if_save_combined_label`: boolean parameter that controls whether to save the combined labels as a .nii.gz file after processing. For example:

    ```
    OUTPUT
    └── case_001
        ├── combined_labels.nii.gz <- (if_save_combined_label)
        └── segmentations
                ├── liver.nii.gz
                ...
                └── veins.nii.gz
    ```