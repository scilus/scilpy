# -*- coding: utf-8 -*-
import numpy as np


def extract_longest_segments_from_profile(strl_indices, atlas_data):
    start_label = None
    end_label = None
    start_idx = None
    end_idx = None

    nb_el_indices = len(strl_indices)
    el_idx = 0
    while start_label is None and el_idx < nb_el_indices:
        if atlas_data[tuple(strl_indices[el_idx])] > 0:
            start_label = atlas_data[tuple(strl_indices[el_idx])]
            start_idx = el_idx
        el_idx += 1

    found_WM = False
    while not found_WM and el_idx < nb_el_indices:
        if atlas_data[tuple(strl_indices[el_idx])] == 0:
            found_WM = True
        el_idx += 1

    if el_idx >= nb_el_indices or not found_WM:
        return []

    el_idx = nb_el_indices - 1
    while end_label is None and el_idx > start_idx:
        if atlas_data[tuple(strl_indices[el_idx])] > 0:
            end_label = atlas_data[tuple(strl_indices[el_idx])]
            end_idx = el_idx
        el_idx -= 1

    if end_label is None or end_idx <= start_idx + 1:
        return []

    return [{'start_label': start_label,
             'start_index': start_idx,
             'end_label': end_label,
             'end_index': end_idx}]


def compute_connectivity(indices, atlas_data, real_labels, segmenting_func):
    connectivity = {k: {lab: [] for lab in real_labels} for k in real_labels}
    for strl_idx, strl_indices in enumerate(indices):
        if (np.array(strl_indices) > atlas_data.shape).any():
            continue
        segments_info = segmenting_func(strl_indices, atlas_data)

        for si in segments_info:
            connectivity[si['start_label']][si['end_label']].append(
                {'strl_idx': strl_idx,
                 'in_idx': si['start_index'],
                 'out_idx': si['end_index']})

    return connectivity
