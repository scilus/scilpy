# -*- coding: utf-8 -*-
import numpy as np


def extract_longest_segments_from_profile(strl_indices, atlas_data):
    """
    For one given streamline, find the labels at both ends.

    Parameters
    ----------
    strl_indices: np.ndarray
        The indices of all voxels traversed by this streamline.
    atlas_data: np.ndarray
        The loaded image containing the labels.

    Returns
    -------
    segments_info: list[dict]
        A list of length 1 with the information dict if , else, an empty list.
    """
    # toDo. background/wm is defined as label 0 in segmenting func, but should
    #  be asked to user.

    start_label = None
    end_label = None
    start_idx = None
    end_idx = None

    nb_underlying_voxels = len(strl_indices)

    # Find the starting point.
    # Advancing if we start in a non-interesting position (label 0, background
    # or WM). Start_label will be the first GM region encountered
    # (corresponding to a label).
    current_vox = 0
    while start_label is None and current_vox < nb_underlying_voxels:
        if atlas_data[tuple(strl_indices[current_vox])] > 0:
            start_label = atlas_data[tuple(strl_indices[current_vox])]
            start_idx = current_vox
        current_vox += 1

    if start_label is None:
        return []

    # Continuing to advance along the streamline. If we do not find a label 0
    # somewhere (WM), this is a weird streamline never leaving GM. Returning []
    found_wm = False
    while not found_wm and current_vox < nb_underlying_voxels:
        if atlas_data[tuple(strl_indices[current_vox])] == 0:
            found_wm = True
        current_vox += 1
    if current_vox >= nb_underlying_voxels or not found_wm:
        return []

    # Find the ending point. As before, moving back as long as we are in a non-
    # interesting position.
    current_vox = nb_underlying_voxels - 1
    while end_label is None and current_vox > start_idx:
        if atlas_data[tuple(strl_indices[current_vox])] > 0:
            end_label = atlas_data[tuple(strl_indices[current_vox])]
            end_idx = current_vox
        current_vox -= 1

    if end_label is None or end_idx <= start_idx + 1:
        return []

    return [{'start_label': start_label,
             'start_index': start_idx,
             'end_label': end_label,
             'end_index': end_idx}]


def compute_connectivity(indices, atlas_data, real_labels, segmenting_func):
    """
    Parameters
    ----------
    indices: ArraySequence
        The list of 3D indices [i, j, k] of all voxels traversed by all
        streamlines. This is the output of our uncompress function.
    atlas_data: np.ndarray
        The loaded image containing the labels.
    real_labels: list
        The list of labels of interest in the image.
    segmenting_func: Callable
        The function used for segmentation.

    Returns
    -------
    connectivity: dict
        A dict containing one key per real_labels (ex, 1, 2) (starting point).

        - The value of connectivity[1] is again a dict with again the
            real_labels as keys.

        - The value of connectivity[1][2] is a list of length n, where n is
            the number of streamlines ending in 1 and finishing in 2. Each
            value is a dict of the following shape:

           >>> 'strl_idx': int --> The idex of the streamline in the raw data.
           >>> 'in_idx:    int -->
           >>> 'out_idx': int  -->
    """
    connectivity = {k: {lab: [] for lab in real_labels} for k in real_labels}

    # toDo. real_labels is not used in segmenting func!
    for strl_idx, strl_vox_indices in enumerate(indices):
        # Managing streamlines out of bound.
        if (np.array(strl_vox_indices) > atlas_data.shape).any():
            continue

        # Finding start_label and end_label.
        segments_info = segmenting_func(strl_vox_indices, atlas_data)
        for si in segments_info:
            connectivity[si['start_label']][si['end_label']].append(
                {'strl_idx': strl_idx,
                 'in_idx': si['start_index'],
                 'out_idx': si['end_index']})

    return connectivity
