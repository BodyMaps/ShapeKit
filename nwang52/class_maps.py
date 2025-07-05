"""
Predefined class maps for organ segmentation
"""

# AbdomenAtlas 1.1 class map (25 organs)
class_map_abdomenatlas_1_1 = {
    1: 'aorta', 
    2: 'gall_bladder', 
    3: 'kidney_left', 
    4: 'kidney_right', 
    5: 'liver', 
    6: 'pancreas', 
    7: 'postcava', 
    8: 'spleen', 
    9: 'stomach', 
    10: 'adrenal_gland_left', 
    11: 'adrenal_gland_right', 
    12: 'bladder', 
    13: 'celiac_trunk', 
    14: 'colon', 
    15: 'duodenum', 
    16: 'esophagus', 
    17: 'femur_left', 
    18: 'femur_right', 
    19: 'hepatic_vessel', 
    20: 'intestine', 
    21: 'lung_left', 
    22: 'lung_right', 
    23: 'portal_vein_and_splenic_vein', 
    24: 'prostate', 
    25: 'rectum'
}

# AbdomenAtlas PANTS class map (28 organs including detailed pancreatic structures)
class_map_abdomenatlas_pants = {
    1: 'adrenal_gland_left',
    2: 'adrenal_gland_right',
    3: 'aorta',
    4: 'bladder',
    5: 'celiac_artery',
    6: 'colon',
    7: 'common_bile_duct',
    8: 'duodenum',
    9: 'femur_left',
    10: 'femur_right',
    11: 'gall_bladder',
    12: 'kidney_left',
    13: 'kidney_right',
    14: 'liver',
    15: 'lung_left',
    16: 'lung_right',
    17: 'pancreas',
    18: 'pancreas_body',
    19: 'pancreas_head',
    20: 'pancreas_tail',
    21: 'pancreatic_duct',
    22: 'postcava',
    23: 'prostate',
    24: 'spleen',
    25: 'stomach',
    26: 'superior_mesenteric_artery',
    27: 'veins',
    28: 'pancreatic_tumor'
}

# Dictionary of all available class maps
available_class_maps = {
    "1.1": class_map_abdomenatlas_1_1,
    "pants": class_map_abdomenatlas_pants
}
