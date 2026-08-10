# Key Functions Guidebook

## remove_small_components
Removes small, disconnected regions from a binary segmentation mask, helping to eliminate noise and improve anatomical plausibility.

<details>
<summary><strong> ❇️ detailed info</strong></summary>

**Signature:**  
`remove_small_components(mask: np.ndarray, threshold: int)`

**Parameters:**
- `mask` (`np.ndarray`): Binary 3D mask.
- `threshold` (`int`): Minimum size (in voxels) for a component to be kept.

**Returns:**
- `np.ndarray`: Cleaned binary mask.

**Example**
```python
cleaned_mask = remove_small_components(mask, threshold=100)
```
</details>

**Applicable to:** `all organs`


## reassign_false_positives
Reassigns false positive regions between anatomically adjacent organs, based on spatial proximity. Improves segmentation specificity by correcting mislabeling.

<details>
<summary><strong> ❇️ detailed info</strong></summary>

**Signature:** 

`reassign_FalsePositives(segmentation_dict: dict, organ_adjacency_map: dict, check_size_threshold: int = 500):`

**Parameters:**
- `segmentation_dict` (`dict`): Mapping of organ names to binary masks.
- `organ_adjacency_map` (`dict`): Defines adjacency between organs.
- `check_size_threshold` (`int`, optional): Minimum component size for reassignment (default: 500).

**Returns:**
- `dict`: Updated segmentation dictionary.

**Example**
```python
segmentation_dict = reassign_FalsePositives(segmentation_dict, organ_adjacency_map)
```
</details>

**Applicable to:** `all organs`


## suppress_non_largest_components_binary

Keeps only the N largest connected components in a binary mask, removing smaller fragments.

<details>
<summary><strong> ❇️ detailed info</strong></summary>

**Signature:**  
`suppress_non_largest_components_binary(mask: np.ndarray, keep_top: int = 2):`

**Parameters:**
- `mask` (`np.ndarray`): Binary 3D mask.
- `keep_top` (`int`): Number of largest components to retain (default: 2).

**Returns:**
- `np.ndarray`: Cleaned binary mask.

**Example**
```python
dominant_mask = suppress_non_largest_components_binary(mask, keep_top=2)
```
</details>

**Applicable to:** `all organs`

## split_right_left

Splits a symmetric organ mask into right and left components along a specified axis.

<details>
<summary><strong> ❇️ detailed info</strong></summary>

**Signature:**  
`split_right_left(mask: np.ndarray, AXIS: int = 0):`

**Parameters:**
- `mask` (`np.ndarray`): Binary 3D mask (e.g., for lungs or kidneys).
- `AXIS` (`int`): Axis along which to perform the split (default: 0).

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (`right_mask`, `left_mask`)

**Example**
```python
right_mask, left_mask = split_right_left(organ_mask, AXIS=0)
```
</details>

**Applicable to:** `adrenal glands `, `lungs`, `kidneys`, `femurs`

## reassign_left_right_based_on_liver
Corrects left and right assignments for organs using the liver as a spatial reference (assumed right-side).

<details>
<summary><strong> ❇️ detailed info</strong></summary>

**Signature:**  
`reassign_left_right_based_on_liver(right_mask: np.ndarray, left_mask: np.ndarray, liver_mask: np.ndarray):`

**Parameters:**
- `right_mask` (`np.ndarray`): Presumed right-side organ mask.
- `left_mask` (`np.ndarray`): Presumed left-side organ mask.
- `liver_mask` (`np.ndarray`): Liver mask (reference for spatial correction).

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (`corrected_right_mask`, `corrected_left_mask`)

**Example**
```python
corrected_right, corrected_left = reassign_left_right_based_on_liver(
    right_mask, left_mask, liver_mask)
```
</details>

**Applicable to:** `adrenal glands`, `lungs`, `kidneys`, `femurs`

## check_organ_location

Removes anatomically implausible voxels from organ segmentations based on spatial relationships with a reference organ (e.g., kidney or liver).
<details>  
<summary><strong> ❇️ detailed info</strong></summary>  

**Signature** 

`check_organ_location(segmentation_dict: dict, organ_mask: np.ndarray, organ_name: str, AXIS_Z: int, reference: str = 'kidney_left'):`

**Parameters:**
- `segmentation_dict` (`dict`): Dictionary mapping organ names to their binary 3D masks.  
- `organ_mask` (`np.ndarray`): Binary mask.  
- `organ_name` (`str`): Name of the organ (used for logging/debugging).  
- `AXIS_Z` (`int`): Axis representing the superior-inferior (head-to-foot) direction.  
- `reference` (`str`): Organ used as an anatomical reference point (default: `'kidney_left'`). Falls back to `'liver'` if unavailable.

**Returns:**
- `np.ndarray`: Corrected binary mask with implausible voxels removed.

**Example**
```python
organ_mask =  check_organ_location(segmentation_dict, organ_mask, 'organ_name', axis_map['z'])
```

</details>

**Applicable to:** `bladder`, `femurs`, `prostate`

## reconstruct_vertebra_instances

Gated cranio-caudal *identity* repair for the vertebral column. When a
contiguous thoracolumbar run has its physical vertebrae split across names (or a
name shared across two bodies), per-label cleanup cannot recover the correct
names. This function rebuilds the physical instances and re-numbers them, but
only when it can do so confidently - a clean or partial scan is returned
unchanged.

<details>
<summary><strong> ❇️ detailed info</strong></summary>

**Signature:**

`reconstruct_vertebra_instances(vol: np.ndarray, min_size: int = 500, affine=None, logger=None, patient_id: str = "")`

**Parameters:**
- `vol` (`np.ndarray`): 3D label volume using the vertebrae ids from `config.yaml` `class_map` (26=`vertebrae_L5` .. 49=`vertebrae_C1`).
- `min_size` (`int`): Components below this voxel count are treated as debris and ignored during clustering.
- `affine`: Optional voxel-to-world affine. ShapeKit supplies the reference image affine so clustering uses world RAS coordinates regardless of voxel orientation or spacing; without it, array axis 2 is used for backwards compatibility.
- `logger`: Optional logger; the chosen action is recorded per case.
- `patient_id` (`str`): Case id used for logging.

**How it works:**
1. Extract per-label connected components (26-connectivity) above `min_size`.
2. Cluster them into 24 physical instances by world-z, behind a triple gate: z-gap (fraction of the median dominant-label spacing), transverse-centroid distance, and original-label adjacency.
3. Re-number the instances cranio-caudally (L5 -> C1).

**Gating (never renumber a healthy/partial scan):**
- `clean_passthrough`: exactly 24 single-component labels in order -> returned unchanged.
- `confident_global_relabel`: excess components **and** at least one gated merge that resolves to exactly 24 instances -> relabelled.
- `fallback_conservative`: any other case (too few/many, no confident merge) -> returned unchanged.

**Returns:**
- `tuple[np.ndarray, str]`: the (possibly repaired) volume and the action taken.

**Example**
```python
repaired, action = reconstruct_vertebra_instances(
    vertebrae_vol, affine=reference_img.affine, logger=logging, patient_id=pid
)
```

**Note:** ShapeKit passes in-memory masks and no CT, so this build omits the
optional CT-derived mid-thoracic re-cut from the original warm-up; it stays
conservative and reproducible without intensity data.

</details>

**Applicable to:** `vertebrae`
