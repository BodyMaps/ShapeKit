class Settings:
    def __init__(
        self,
        input_folder: str = None,
        output_folder: str = None,
        cpu_count: int = 4,
        start_idx: int = 0,
        end_idx: int = -1,
    ):
        """
        Postprocessing abdominal CT masks.

        Args:
            input_folder: Directory containing .nii.gz files to be postprocessed.
            output_folder: Directory to save the postprocessed files.
            cpu_count: Number of parallel jobs to run for postprocessing.
            start_idx: Start index (inclusive) for selecting a subset of source directories.
            end_idx: End index (exclusive) for selecting a subset of source directories. If -1, all files after start index (inclusive) are processed.
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

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.cpu_count = cpu_count
        self.start_idx = start_idx
        self.end_idx = end_idx
