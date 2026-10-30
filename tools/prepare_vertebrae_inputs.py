"""Convert standalone internal 0..24 combined masks to ShapeKit input layout.

The original masks are read and never modified. Do not pass ShapeKit 26..49
combined maps or masks already processed by a different backend.
"""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np

NAMES = ([f'vertebrae_{n}' for n in ('L5','L4','L3','L2','L1')]
         + [f'vertebrae_T{i}' for i in range(12, 0, -1)]
         + [f'vertebrae_C{i}' for i in range(7, 0, -1)])


def prepare(input_root, output_root, selected=None):
    if input_root.resolve() == output_root.resolve() or input_root.resolve() in output_root.resolve().parents or output_root.resolve() in input_root.resolve().parents:
        raise ValueError('Input and output roots must not overlap')
    folders = sorted(input_root.glob('BDMAP_*'))
    if selected:
        folders = [folder for folder in folders if folder.name in selected]
        if {f.name for f in folders} != set(selected):
            raise ValueError('One or more selected cases were not found')
    if not folders:
        raise FileNotFoundError('No BDMAP_* case directories found')
    for folder in folders:
        src = folder / 'combined_labels.nii.gz'
        if not src.is_file():
            raise FileNotFoundError(str(src))
        image = nib.load(str(src))
        data = np.asarray(image.dataobj)
        if data.ndim != 3 or not np.all(np.isfinite(data)) or not np.all(data == np.round(data)) or np.any((data < 0) | (data > 24)):
            raise ValueError(f'{src}: expected integer internal labels 0..24')
        stage = output_root / folder.name / 'segmentations'
        if stage.exists():
            raise FileExistsError(f'Refusing to overwrite existing segmentations: {stage}')
        stage.mkdir(parents=True)
        for label, name in enumerate(NAMES, 1):
            binary = (data == label).astype(np.uint8)
            header = image.header.copy()
            header.set_data_dtype(np.uint8)
            out = nib.Nifti1Image(binary, image.affine, header=header)
            out.set_qform(image.get_qform(), int(image.header['qform_code']))
            out.set_sform(image.get_sform(), int(image.header['sform_code']))
            nib.save(out, str(stage / f'{name}.nii.gz'))
        print(f'[Prepare] {folder.name}: wrote 24 masks into {stage}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=Path, required=True)
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    prepare(args.input_root, args.output_root, args.case)
