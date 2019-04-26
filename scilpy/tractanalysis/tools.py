# -*- coding: utf-8 -*-
#
# from __future__ import division
#

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
    ray = ray / np.linalg.norm(ray)

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
#
#
# def intersects_two_rois(roi_data_1, roi_data_2, voxel_map):
#     entry_found = False
#     exit_found = False
#     went_out_of_exit = False
#     exit_roi_data = None
#     in_strl_idx = None
#     out_strl_idx = None
#
#     # TODO simplify
#     strl_indices = voxel_map
#
#     logging.debug(strl_indices)
#     for idx, point in enumerate(strl_indices):
#         # logging.debug("Point: {}".format(point))
#         if entry_found and exit_found:
#             # Still add points that are in exit roi, to mimic entry ROI
#             # This will need to be modified to correctly handle continuation
#             if exit_roi_data[tuple(point)] > 0:
#                 if not went_out_of_exit:
#                     out_strl_idx = idx
#             else:
#                 went_out_of_exit = True
#         elif entry_found and not exit_found:
#             # If we reached the exit ROI
#             if exit_roi_data[tuple(point)] > 0:
#                 exit_found = True
#                 out_strl_idx = idx
#         elif not entry_found:
#             # Check if we are in one of ROIs
#             if roi_data_1[tuple(point)] > 0 or roi_data_2[tuple(point)] > 0:
#                 entry_found = True
#                 in_strl_idx = idx
#                 if roi_data_1[tuple(point)] > 0:
#                     exit_roi_data = roi_data_2
#                 else:
#                     exit_roi_data = roi_data_1
#
#     return in_strl_idx, out_strl_idx
#
#
# def intersects_two_rois_atlas(atlas_data, voxel_map,
#                               allow_self_connection=True,
#                               minimize_self_connections=True):
#     entry_found = False
#     exit_found = False
#
#     entry_label = None
#     exit_label = None
#     went_out_of_entry = False
#     reentered_entry = False
#
#     in_strl_idx = None
#     out_strl_idx = None
#
#     # TODO simplify
#     strl_indices = voxel_map
#
#     for idx, point in enumerate(strl_indices):
#         # logging.debug("Point: {}".format(point))
#         if entry_found and not exit_found:
#             if atlas_data[tuple(point)] == 0:
#                 went_out_of_entry = True
#             elif atlas_data[tuple(point)] == entry_label:
#                 if went_out_of_entry:
#                     if not allow_self_connection:
#                         continue
#                     else:
#                         reentered_entry = True
#                         if not minimize_self_connections:
#                             exit_label = entry_label
#                             exit_found = True
#                             out_strl_idx = idx
#             else:
#                 exit_found = True
#                 exit_label = atlas_data[tuple(point)]
#                 out_strl_idx = idx
#         elif not entry_found:
#             # Check if we are in one of ROIs
#             if atlas_data[tuple(point)] > 0:
#                 entry_found = True
#                 entry_label = atlas_data[tuple(point)]
#                 in_strl_idx = idx
#
#     # Can happen when allowing self connections but minimizing
#     if entry_label is not None and exit_label is None:
#         if allow_self_connection and reentered_entry:
#             exit_label == entry_label
#             # TODO missing the index
#
#     return entry_label, exit_label, in_strl_idx, out_strl_idx


def compute_streamline_segment(orig_strl, inter_vox, in_vox_idx, out_vox_idx,
                               points_to_indices):
    additional_start_pt = None
    additional_exit_pt = None
    nb_points = 0

    # Check if the indexed voxel contains a real streamline point
    in_strl_point = get_streamline_pt_index(points_to_indices,
                                            in_vox_idx)

    if in_strl_point is None:
        #logging.debug("No direct mapping to in streamline point found.")
        #logging.debug("  Looking for next valid point")

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
        #logging.debug("No direct mapping to streamline exit point found.")
        #logging.debug("  Looking for previous valid point")

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
        #logging.debug("  adding artifical entry point: {}".format(
        #              additional_start_pt))
        segment[0] = additional_start_pt
        at_point += 1

    if exit_strl_point >= in_strl_point:
        #logging.debug("  adding points [{}:{}] from orig strl".format(
        #              in_strl_point, exit_strl_point + 1))

        # Note: this works correctly even in the case where the "previous"
        # point is the same or lower than the entry point, because of
        # numpy indexing
        ##print(in_strl_point, exit_strl_point)
        segment[at_point:at_point + nb_points_orig_strl] = \
            orig_strl[in_strl_point:exit_strl_point + 1]
        at_point += nb_points_orig_strl
    #else:
        #logging.debug("  exit point was found before entry point. "
        #              "Not inserting from orig strl.")


    if additional_exit_pt is not None:
        #logging.debug("  adding artifical exit point: {}".format(
        #              additional_exit_pt))
        segment[at_point] = additional_exit_pt
        at_point += 1

    #logging.debug("    new segment: {}".format(segment))
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

    # TODO check self connections
    if end_label is None or end_idx <= start_idx + 1:
        return []

    return [{'start_label': start_label,
             'start_index': start_idx,
             'end_label': end_label,
             'end_index': end_idx}]


# TODO self connect
def compute_connectivity(indices, atlas_data,
                         segmenting_func,
                         allow_self_connection,
                         minimize_self_connection):
    atlas_data = atlas_data.astype(np.int32)
    real_labels = np.unique(atlas_data)

    connectivity = {k: {lab: [] for lab in real_labels} for k in real_labels}

    for strl_idx, strl_indices in enumerate(indices):
        segments_info = segmenting_func(strl_indices, atlas_data)

        for si in segments_info:
            connectivity[si['start_label']][si['end_label']].append(
                {'strl_idx': strl_idx,
                 'in_idx': si['start_index'],
                 'out_idx': si['end_index']})

    return connectivity
