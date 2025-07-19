import numpy as np


class Settings:
    def __init__(
        self,
        input_folder: str = None,
        output_folder: str = None,
        cpu_count: int = 4,
        start_idx: int = 0,
        end_idx: int = -1,
        save_combined_labels: bool = False,
    ):
        """
        Postprocessing abdominal CT masks.

        Args:
            input_folder: Directory containing .nii.gz files to be postprocessed.
            output_folder: Directory to save the postprocessed files.
            cpu_count: Number of parallel jobs to run for postprocessing.
            start_idx: Start index (inclusive) for selecting a subset of source directories.
            end_idx: End index (exclusive) for selecting a subset of source directories. If -1, all files after start index (inclusive) are processed.
            save_combined_labels: If True, saves the combined labels in the output folder.
        """
        # sanity checks
        assert input_folder is not None, "input_folder must be specified"
        assert output_folder is not None, "output_folder must be specified"
        assert cpu_count > 0, "cpu_count must be a positive integer"
        if end_idx == -1:
            assert start_idx >= 0, "start_idx must be non-negative"
        else:
            assert start_idx >= 0, "start_idx must be non-negative"
            assert end_idx > start_idx, "end_idx must be greater than start_idx"
        assert isinstance(
            save_combined_labels, bool
        ), "save_combined_labels must be a boolean value"

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.cpu_count = cpu_count
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.save_combined_labels = save_combined_labels


class MaskCropper:
    def __init__(self):
        self.original_shape = None
        self.bbox = None

    def get_tightest_bbox(self, seg_mask):
        """Get tight bounding box of a binary mask."""
        masks = [i for i in seg_mask.values()]
        combined_mask = np.logical_or.reduce(masks)

        self.original_shape = combined_mask.shape

        coords = np.argwhere(combined_mask)

        if len(coords) != 0:
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0) + 1
            self.bbox = (z_min, z_max, y_min, y_max, x_min, x_max)

    def crop(self, seg_mask):
        """Crop mask using the bounding box."""
        assert self.bbox is not None

        z_min, z_max, y_min, y_max, x_min, x_max = self.bbox
        for label, mask in seg_mask.items():
            seg_mask[label] = mask[z_min:z_max, y_min:y_max, x_min:x_max]
        return seg_mask

    def restore(self, seg_mask):
        """Restore the cropped mask to original shape with padding."""
        assert self.bbox is not None

        z_min, z_max, y_min, y_max, x_min, x_max = self.bbox
        for label, mask in seg_mask.items():
            full_mask = np.zeros(self.original_shape, dtype=mask.dtype)
            full_mask[z_min:z_max, y_min:y_max, x_min:x_max] = mask
            seg_mask[label] = full_mask
        return seg_mask
