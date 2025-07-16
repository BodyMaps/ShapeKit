class Settings:
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        n_jobs: int = 4,
        axcodes: str = "auto",
        start_idx: int = 0,
        end_idx: int = -1,
    ):
        """
        Postprocessing abdominal CT masks.

        Args:
            source_dir: Directory containing .nii.gz files to be postprocessed.
            target_dir: Directory to save the postprocessed files.
            n_jobs: Number of parallel jobs to run for postprocessing.
            axcodes: Axes codes for the orientation of the images. Can be a tuple or 'auto'.
                    If 'auto', it will be in the order of (R, A, S).
            start_idx: Start index (inclusive) for selecting a subset of source directories.
            end_idx: End index (exclusive) for selecting a subset of source directories. If -1, all files after start index (inclusive) are processed.
        """
        # sanity checks
        assert isinstance(axcodes, (tuple, str)), "axcodes must be a tuple or string"
        assert n_jobs > 0, "n_jobs must be a positive integer"
        if end_idx == -1:
            assert start_idx >= 0, "start_idx must be non-negative"
        else:
            assert start_idx >= 0, "start_idx must be non-negative"
            assert end_idx > start_idx, "end_idx must be greater than start_idx"

        self.source_dir = source_dir
        self.target_dir = target_dir
        self.n_jobs = n_jobs
        self.axcodes = axcodes
        self.start_idx = start_idx
        self.end_idx = end_idx
