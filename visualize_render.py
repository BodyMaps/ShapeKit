import numpy as np
import nibabel as nib
import pyvista as pv
import numpy as np
import os
import matplotlib.colors as mcolors




group_dict = {
    '1': ['lung_left', 'lung_right', 'esophagus', 'liver', 'gall_bladder', 'hepatic_vessel', 'portal_vein_and_splenic_vein', 'kidney_left', 'kidney_right', 'adrenal_gland_left', 'adrenal_gland_right', 'postcava'],
    '2': ['stomach', 'pancreas', 'duodenum', 'colon', 'intestine', 'rectum'],
    '3': ['aorta', 'celiac_trunk', 'bladder', 'prostate', 'femur_left', 'femur_right']
}



def get_custom_cmap(n_colors):
    base_colors = [
        "#87FF97",
        "#FF3C3C",  
        "#96BDE4",  
        "#8B7A2E",  
        "#7258B1",  
        "#116D3F",  
        "#AF7798", 
        "#805656" 
    ]
    # Repeat if more organs than colors
    colors = (base_colors * ((n_colors + len(base_colors) - 1) // len(base_colors)))[:n_colors]
    return mcolors.ListedColormap(colors)


def get_labeled_group_mask(group_name, group_dict, location):
    """
    Loads binary masks from disk and combines them into a single labeled mask.
    Each organ in the group is assigned a unique label starting from 1.

    Parameters:
    - group_name (str): Group key from group_dict (e.g., '1').
    - group_dict (dict): Dictionary mapping group names to lists of organ names.
    - location (str): Path to the root directory containing organ masks in 'segmentations/' subfolder.

    Returns:
    - labeled_mask (ndarray): 3D array with unique integer labels for each organ.
    """
    organ_list = group_dict[group_name]
    
    # Load the first mask to get shape
    first_path = os.path.join(location, 'combined_labels.nii.gz')
    shape = nib.load(first_path).get_fdata().shape
    labeled_mask = np.zeros(shape, dtype=np.uint8)

    for idx, organ in enumerate(organ_list, start=1):
        try:
            organ_path = os.path.join(location, 'segmentations', f'{organ}.nii.gz')
            nii_img = nib.load(organ_path)
            binary_mask = nii_img.get_fdata() > 0
            labeled_mask[binary_mask] = np.maximum(labeled_mask[binary_mask], idx)

            print(f"Loaded {organ}, \tassigned idx: {idx}, \tvolumn: {np.sum(binary_mask)}")
        except:
            print(f"Organ {organ} does not exist, pass ...")

    return labeled_mask


def render_3d_fast(segmentation_mask, save_path='u', zoom_factor=1.5, AXIS_z=2, z_reverse_bool=False, cmap=None):
    """
    Fast 3D rendering using pyvista for large segmentation masks.
    """

    if z_reverse_bool:
        segmentation_mask = np.flip(segmentation_mask, axis=AXIS_z)

    # Create a volume from the numpy array
    grid = pv.wrap(segmentation_mask.astype(np.uint8))

    # Plot with threshold to only show segmented voxels
    plotter = pv.Plotter( window_size=(500, 500))
    plotter.add_volume(grid, cmap=cmap, opacity="linear", shade=True, show_scalar_bar=False)
    plotter.view_vector((0, -1, 0))  # View along Z axis
    plotter.show_axes() 
    plotter.camera.zoom(zoom_factor) 

    light = pv.Light(
    light_type='headlight',  # Follows the camera
    intensity=1.5,           # Stronger than default (1.0)
    color='white'
    )
    plotter.add_light(light)
    plotter.set_background("black")

    save_path += '.svg'
    plotter.save_graphic(save_path)
    plotter.show()



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--location", type=str, default="NV/BDMAP_00013500")
parser.add_argument("--group", type=str, default=1)
parser.add_argument("--save_name", type=str)

args = parser.parse_args()
location = args.location
# organ = args.organ
group = args.group
save_name = args.save_name

if __name__ == '__main__':
    
    mask = get_labeled_group_mask(group, group_dict, location)
    organ_count = len(group_dict[group])
    cmap = get_custom_cmap(organ_count)

    render_3d_fast(
        segmentation_mask=mask, 
        zoom_factor=1.5,
        AXIS_z=2, 
        save_path=f'{str(save_name)}_group{group.upper()}', 
        z_reverse_bool=True,
        cmap=cmap,
        )