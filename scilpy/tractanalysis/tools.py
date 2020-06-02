# -*- coding: utf-8 -*-

import numpy as np

from scilpy.tractanalysis.quick_tools import (get_next_real_point,
                                              get_previous_real_point)


def get_streamline_pt_index(points_to_index, vox_index, from_start=True):
    cur_idx = np.where(points_to_index == vox_index)

    if not len(cur_idx[0]):
        return None

    if from_start:
        idx_to_take = 0
    else:
        idx_to_take = -1

    return cur_idx[0][idx_to_take]


def get_point_on_line(first_point, second_point, vox_lower_corner):
    # To manage the case where there is no real streamline point in an
    # intersected voxel, we need to generate an artificial point.
    # We use line / cube intersections as presented in
    # Physically Based Rendering, Second edition, pp. 192-195
    # Some simplifications are made since we are sure that an intersection
    # exists (else this function would not have been called).
    ray = second_point - first_point
    ray /= np.linalg.norm(ray)

    corners = np.array([vox_lower_corner, vox_lower_corner + 1])

    t0 = 0
    t1 = np.inf
    for i in range(3):
        if ray[i] != 0.:
            inv_ray = 1. / ray[i]
            v0 = (corners[0, i] - first_point[i]) * inv_ray
            v1 = (corners[1, i] - first_point[i]) * inv_ray
            t0 = max(t0, min(v0, v1))
            t1 = min(t1, max(v0, v1))

    return first_point + ray * (t0 + t1) / 2.


def compute_streamline_segment(orig_strl, inter_vox, in_vox_idx, out_vox_idx,
                               points_to_indices):
    additional_start_pt = None
    additional_exit_pt = None
    nb_points = 0

    # Check if the indexed voxel contains a real streamline point
    in_strl_point = get_streamline_pt_index(points_to_indices,
                                            in_vox_idx)

    if in_strl_point is None:
        # Find the next real streamline point
        in_strl_point = get_next_real_point(points_to_indices, in_vox_idx)

        additional_start_pt = get_point_on_line(orig_strl[in_strl_point - 1],
                                                orig_strl[in_strl_point],
                                                inter_vox[in_vox_idx])
        nb_points += 1

    # Generate point for the current voxel
    exit_strl_point = get_streamline_pt_index(points_to_indices,
                                              out_vox_idx,
                                              from_start=False)

    if exit_strl_point is None:
        # Find the previous real streamline point
        exit_strl_point = get_previous_real_point(points_to_indices,
                                                  out_vox_idx)

        additional_exit_pt = get_point_on_line(orig_strl[exit_strl_point],
                                               orig_strl[exit_strl_point + 1],
                                               inter_vox[out_vox_idx])
        nb_points += 1

    if exit_strl_point >= in_strl_point:
        nb_points_orig_strl = exit_strl_point - in_strl_point + 1
        nb_points += nb_points_orig_strl

    segment = np.zeros((nb_points, 3))
    at_point = 0

    if additional_start_pt is not None:
        segment[0] = additional_start_pt
        at_point += 1

    if exit_strl_point >= in_strl_point:
        # Note: this works correctly even in the case where the "previous"
        # point is the same or lower than the entry point, because of
        # numpy indexing
        segment[at_point:at_point + nb_points_orig_strl] = \
            orig_strl[in_strl_point:exit_strl_point + 1]
        at_point += nb_points_orig_strl

    if additional_exit_pt is not None:
        segment[at_point] = additional_exit_pt
        at_point += 1

    return segment


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
    atlas_data = atlas_data.astype(np.int32)

    connectivity = {k: {lab: [] for lab in real_labels} for k in real_labels}

    for strl_idx, strl_indices in enumerate(indices):
        segments_info = segmenting_func(strl_indices, atlas_data)

        for si in segments_info:
            connectivity[si['start_label']][si['end_label']].append(
                {'strl_idx': strl_idx,
                 'in_idx': si['start_index'],
                 'out_idx': si['end_index']})

    return connectivity
