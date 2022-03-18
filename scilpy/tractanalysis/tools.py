# -*- coding: utf-8 -*-

from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np

from scilpy.tracking.tools import filter_streamlines_by_length
from scilpy.tractanalysis.quick_tools import (get_next_real_point,
                                              get_previous_real_point)
from scilpy.tractanalysis.reproducibility_measures import get_endpoints_density_map
from scilpy.tractanalysis.scoring import split_heads_tails_kmeans
from scilpy.tractanalysis.uncompress import uncompress


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


def cut_outside_of_mask_streamlines(sft, binary_mask, min_len=0):
    """ Cut streamlines so their longest segment are within the bounding box
    or a binary mask.
    This function erases the data_per_point and data_per_streamline.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to cut streamlines (using a single mask with 1 entities) from.
    binary_mask: np.ndarray
        Boolean array representing the region (must contain 1 entities)
    min_len: float
        Minimum length from the resulting streamlines.

    Returns
    -------
    new_sft : StatefulTractogram
        New object with the streamlines trimmed within the mask.
    """
    sft.to_vox()
    sft.to_corner()
    streamlines = sft.streamlines

    new_streamlines = []
    for _, streamline in enumerate(streamlines):
        entry_found = False
        last_success = 0
        curr_len = 0
        longest_seq = (0, 0)
        for ind, pos in enumerate(streamline):
            pos = tuple(pos.astype(np.int16))
            if binary_mask[pos]:
                if not entry_found:
                    entry_found = True
                    last_success = ind
                    curr_len = 0
                else:
                    curr_len += 1
                    if curr_len > longest_seq[1] - longest_seq[0]:
                        longest_seq = (last_success, ind+1)
            else:
                if entry_found:
                    entry_found = False
                    if curr_len > longest_seq[1] - longest_seq[0]:
                        longest_seq = (last_success, ind)
                        curr_len = 0
        # print(longest_seq)
        if longest_seq[1] != 0:
            new_streamlines.append(streamline[longest_seq[0]:longest_seq[1]])

    new_sft = StatefulTractogram.from_sft(new_streamlines, sft)
    return filter_streamlines_by_length(new_sft, min_length=min_len)


def cut_between_masks_streamlines(sft, binary_mask, min_len=0):
    """ Cut streamlines so their segment are within the bounding box
    or going from binary mask #1 to binary mask #2.
    This function erases the data_per_point and data_per_streamline.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to cut streamlines (using a single mask with 2 entities) from.
    binary_mask: np.ndarray
        Boolean array representing the region (must contain 2 entities)
    min_len: float
        Minimum length from the resulting streamlines.
    Returns
    -------
    new_sft : StatefulTractogram
        New object with the streamlines trimmed within the masks.
    """
    sft.to_vox()
    sft.to_corner()
    streamlines = sft.streamlines

    density = get_endpoints_density_map(streamlines, binary_mask.shape)
    density[density > 0] = 1
    density[binary_mask == 0] = 0

    roi_data_1, roi_data_2 = split_heads_tails_kmeans(binary_mask)

    new_streamlines = []
    (indices, points_to_idx) = uncompress(streamlines, return_mapping=True)

    for strl_idx, strl in enumerate(streamlines):
        strl_indices = indices[strl_idx]

        in_strl_idx, out_strl_idx = intersects_two_rois(roi_data_1,
                                                        roi_data_2,
                                                        strl_indices)

        if in_strl_idx is not None and out_strl_idx is not None:
            points_to_indices = points_to_idx[strl_idx]
            tmp = compute_streamline_segment(strl, strl_indices, in_strl_idx,
                                             out_strl_idx, points_to_indices)
            new_streamlines.append(tmp)

    new_sft = StatefulTractogram.from_sft(new_streamlines, sft)
    return filter_streamlines_by_length(new_sft, min_length=min_len)


def intersects_two_rois(roi_data_1, roi_data_2, strl_indices):
    """ Cut streamlines so their longest segment are within the bounding box
    or a binary mask.
    This function keeps the data_per_point and data_per_streamline.

    Parameters
    ----------
    roi_data_1: np.ndarray
        Boolean array representing the region #1
    roi_data_2: np.ndarray
        Boolean array representing the region #2
    strl_indices: list of tuple (3xint)
        3D indices of the voxel intersected by the streamline

    Returns
    -------
    in_strl_idx : int
        index of the first point (of the streamline) to be in the masks
    out_strl_idx : int
        index of the last point (of the streamline) to be in the masks
    """
    entry_found = False
    exit_found = False
    went_out_of_exit = False
    exit_roi_data = None
    in_strl_idx = None
    out_strl_idx = None

    for idx, point in enumerate(strl_indices):
        if entry_found and exit_found:
            # Still add points that are in exit roi, to mimic entry ROI
            # This will need to be modified to correctly handle continuation
            if exit_roi_data[tuple(point)] > 0:
                if not went_out_of_exit:
                    out_strl_idx = idx
            else:
                went_out_of_exit = True
        elif entry_found and not exit_found:
            # If we reached the exit ROI
            if exit_roi_data[tuple(point)] > 0:
                exit_found = True
                out_strl_idx = idx
        elif not entry_found:
            # Check if we are in one of ROIs
            if roi_data_1[tuple(point)] > 0 or roi_data_2[tuple(point)] > 0:
                entry_found = True
                in_strl_idx = idx
                if roi_data_1[tuple(point)] > 0:
                    exit_roi_data = roi_data_2
                else:
                    exit_roi_data = roi_data_1

    return in_strl_idx, out_strl_idx
