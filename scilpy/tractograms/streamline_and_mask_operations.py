# -*- coding: utf-8 -*-
import logging

import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import set_number_of_points

from scipy.ndimage import map_coordinates

from scilpy.tractograms.uncompress import uncompress

from scilpy.image.utils import split_mask_blobs_kmeans
from scilpy.tractanalysis.quick_tools import (get_next_real_point,
                                              get_previous_real_point)
from scilpy.tractograms.streamline_operations import \
    filter_streamlines_by_length, _get_point_on_line, _get_streamline_pt_index


def get_endpoints_density_map(sft, point_to_select=1):
    """
    Compute an endpoints density map, supports selecting more than one points
    at each end.

    Parameters
    ----------
    sft: StatefulTractogram
        The streamlines to compute endpoints density from.
    point_to_select: int
        Instead of computing the density based on the first and last points,
        select more than one at each end.

    Returns
    -------
    np.ndarray: A np.ndarray where voxel values represent the density of
        endpoints.
    """

    endpoints_map_head, endpoints_map_tail = \
        get_head_tail_density_maps(sft, point_to_select)
    return endpoints_map_head + endpoints_map_tail


def get_head_tail_density_maps(sft, point_to_select=1):
    """
    Compute two separate endpoints density maps for the head and tail of
    a list of streamlines.

    Parameters
    ----------
    sft: StatefulTractogram
        The streamlines to compute endpoints density from.
    point_to_select: int
        Instead of computing the density based on the first and last points,
        select more than one at each end. To support compressed streamlines,
        a resampling to 0.5mm per segment is performed.

    Returns
    -------
    A tuple containing:
    - np.ndarray: A np.ndarray where voxel values represent the density of
        head endpoints.
    - np.ndarray: A np.ndarray where voxel values represent the density of
        tail endpoints.
    """

    sft.to_vox()
    sft.to_corner()

    dimensions = sft.dimensions
    streamlines = sft.streamlines

    endpoints_map_head = np.zeros(dimensions)
    endpoints_map_tail = np.zeros(dimensions)

    # A possible optimization would be to compute all coordinates first
    # and then do the np.add.at only once.
    for streamline in streamlines:

        # Resample the streamline to make sure we have enough points
        nb_point = max(len(streamline), point_to_select*2)
        streamline = set_number_of_points(streamline, nb_point)

        # Get the head and tail coordinates
        points_list_head = streamline[0:point_to_select, :]
        points_list_tail = streamline[-point_to_select:, :]

        # Convert the points to indices by rounding them and clipping them
        head_indices = np.clip(
            points_list_head, 0, np.asarray(dimensions) - 1).astype(int).T
        tail_indices = np.clip(
            points_list_tail, 0, np.asarray(dimensions) - 1).astype(int).T

        # Add the points to the endpoints map
        # Note: np.add.at is used to support duplicate points
        np.add.at(endpoints_map_tail, tuple(tail_indices), 1)
        np.add.at(endpoints_map_head, tuple(head_indices), 1)

    return endpoints_map_head, endpoints_map_tail


def cut_outside_of_mask_streamlines(sft, binary_mask, min_len=0):
    """
    Cut streamlines so their longest segment are within the bounding box or a
    binary mask.

    This function erases the data_per_point.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to cut streamlines (using a single mask with 1 entity) from.
    binary_mask: np.ndarray
        Boolean array representing the region (must contain 1 entity)
    min_len: float
        Minimum length from the resulting streamlines.

    Returns
    -------
    new_sft : StatefulTractogram
        New object with the streamlines trimmed within the mask.
    """
    orig_space = sft.space
    orig_origin = sft.origin
    sft.to_vox()
    sft.to_corner()

    # Cut streamlines within the mask and return the new streamlines
    # New endpoints may be generated
    logging.info("Cutting streamlines. Data_per_point will not be kept.")
    new_streamlines, kept_idx = _cut_streamlines_with_masks(
        sft.streamlines, binary_mask, binary_mask)
    if len(kept_idx) != len(sft.streamlines):
        logging.info("{}/{} streamlines were kept."
                     .format(len(kept_idx), len(sft.streamlines)))

    new_sft = StatefulTractogram.from_sft(
        new_streamlines, sft,
        data_per_streamline=sft.data_per_streamline[kept_idx])
    new_sft.to_space(orig_space)
    new_sft.to_origin(orig_origin)
    return filter_streamlines_by_length(new_sft, min_length=min_len)


def cut_between_mask_two_blobs_streamlines(sft, binary_mask_1,
                                           binary_mask_2=None,
                                           min_len=0):
    """
    Cut streamlines so their segment are going from blob #1 to blob #2 in a
    binary mask. This function presumes strictly two blobs are present in the
    mask.

    This function erases the data_per_point.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to cut streamlines (using a single mask with 2 entities) from.
    binary_mask_1: np.ndarray
        Boolean array representing the region (must contain 2 entities)
    binary_mask_2: np.ndarray
        Boolean array representing the region (must contain 2 entities)
    min_len: float
        Minimum length from the resulting streamlines.

    Returns
    -------
    new_sft : StatefulTractogram
        New object with the streamlines trimmed within the masks.
    """
    orig_space = sft.space
    orig_origin = sft.origin
    sft.to_vox()
    sft.to_corner()

    if binary_mask_2:
        roi_data_1 = binary_mask_1
        roi_data_2 = binary_mask_2
    else:
        # Split head and tail from mask
        roi_data_1, roi_data_2 = split_mask_blobs_kmeans(
            binary_mask_1, nb_clusters=2)

    # Cut streamlines with the masks and return the new streamlines
    # New endpoints may be generated
    logging.info("Cutting streamlines. Data_per_point will not be kept.")
    new_streamlines, kept_idx = _cut_streamlines_with_masks(
        sft.streamlines, roi_data_1, roi_data_2)
    if len(kept_idx) != len(sft.streamlines):
        logging.info("{}/{} streamlines were kept."
                     .format(len(kept_idx), len(sft.streamlines)))

    new_sft = StatefulTractogram.from_sft(
        new_streamlines, sft,
        data_per_streamline=sft.data_per_streamline[kept_idx])
    new_sft.to_space(orig_space)
    new_sft.to_origin(orig_origin)
    return filter_streamlines_by_length(new_sft, min_length=min_len)


def _cut_streamlines_with_masks(streamlines, roi_data_1, roi_data_2):
    """
    Cut streamlines so their segment are going from binary mask #1 to binary
    mask #2. New endpoints may be generated to maximize the streamline length
    within the masks.
    """
    new_streamlines = []
    kept_idx = []
    # Get the indices of the "voxels" intersected by the streamlines and the
    # mapping from points to indices.
    (indices, points_to_idx) = uncompress(streamlines, return_mapping=True)

    if len(streamlines[0]) != len(points_to_idx[0]):
        raise ValueError("Error in the uncompress function. Try running the "
                         "scil_tractogram_remove_invalid.py script with the \n"
                         "--remove_single_point and --remove_overlapping_points"
                         " options.")

    for strl_idx, strl in enumerate(streamlines):
        # The "voxels" intersected by the current streamline
        strl_indices = indices[strl_idx]
        # Find the first and last "voxels" of the streamline that are in the
        # ROIs
        in_strl_idx, out_strl_idx = _intersects_two_rois(roi_data_1,
                                                         roi_data_2,
                                                         strl_indices)
        # If the streamline intersects both ROIs
        if in_strl_idx is not None and out_strl_idx is not None:
            points_to_indices = points_to_idx[strl_idx]
            # Compute the new streamline by keeping only the segment between
            # the two ROIs
            cut_strl = compute_streamline_segment(strl, strl_indices,
                                                  in_strl_idx, out_strl_idx,
                                                  points_to_indices)
            # Add the new streamline to the sft
            new_streamlines.append(cut_strl)
            kept_idx.append(strl_idx)

    return new_streamlines, kept_idx


def _get_longest_streamline_segment_in_roi(all_strl_indices):
    """ Get the longest segment of a streamline that is in a ROI
    using the indices of the voxels intersected by the streamline.

    Parameters
    ----------
    strl_indices: list of streamline indices (N)

    Returns
    -------
    in_strl_idx : int
        Consectutive indices of the streamline that are in the ROI
    """
    # If there is only one index, its likely invalid
    if len(all_strl_indices) == 1:
        return [None]

    # Find the gradient of the indices of the voxels intersecting with
    # the ROIs
    strl_indices_grad = np.gradient(all_strl_indices)
    split_pos = np.where(strl_indices_grad != 1)[0]

    # Covers weird cases where there is only non consecutive indices
    if len(strl_indices_grad) == len(split_pos) + 1:
        return [None]

    # Split the indices of the voxels intersecting with the ROIs into
    # segments where the gradient is 1 (i.e a chunk of consecutive indices)
    strl_indices_split = np.split(all_strl_indices, split_pos)

    # Find the length of each segment
    lens_strl_indices_split = [len(x) for x in strl_indices_split]
    # Keep the segment with the longest length
    strl_indices = strl_indices_split[
        np.argmax(lens_strl_indices_split)]

    return strl_indices


def _intersects_two_rois(roi_data_1, roi_data_2, strl_indices):
    """ Find the first and last "voxels" of the streamline that are in the
    ROIs.

    Parameters
    ----------
    roi_data_1: np.ndarray
       Boolean array representing the region #1
    roi_data_2: np.ndarray
        Boolean array representing the region #2
    strl_indices: list of tuple (N, 3)
        3D indices of the voxels intersected by the streamline

    Returns
    -------
    in_strl_idx : int
        index of the first point (of the streamline) to be in the masks
    out_strl_idx : int
        index of the last point (of the streamline) to be in the masks
    """

    # Find all the points of the streamline that are in the ROIs
    roi_data_1_intersect = map_coordinates(
        roi_data_1, strl_indices.T, order=0, mode='constant', cval=0)
    roi_data_2_intersect = map_coordinates(
        roi_data_2, strl_indices.T, order=0, mode='constant', cval=0)

    # Get the indices of the voxels intersecting with the ROIs
    in_strl_indices = np.argwhere(roi_data_1_intersect).squeeze(-1)
    out_strl_indices = np.argwhere(roi_data_2_intersect).squeeze(-1)

    # If there are no points in the ROIs, return None
    if len(in_strl_indices) == 0:
        in_strl_indices = [None]
    else:
        # Get the longest segment of the streamline that is in the ROI
        in_strl_indices = _get_longest_streamline_segment_in_roi(
            in_strl_indices)

    if len(out_strl_indices) == 0:
        out_strl_indices = [None]
    else:
        out_strl_indices = _get_longest_streamline_segment_in_roi(
            out_strl_indices)

    # If the entry point is after the exit point, swap them
    if in_strl_indices[0] is not None and out_strl_indices[0] is not None \
       and min(in_strl_indices) > min(out_strl_indices):
        in_strl_indices, out_strl_indices = out_strl_indices, in_strl_indices

    # Get the index of the first and last "voxels" of the streamline that are
    # in the ROIs
    in_strl_idx = in_strl_indices[0]
    out_strl_idx = out_strl_indices[-1]

    return in_strl_idx, out_strl_idx


def compute_streamline_segment(orig_strl, inter_vox, in_vox_idx, out_vox_idx,
                               points_to_indices):
    """ Compute the segment of a streamline that is in a given ROI or
    between two ROIs.

    If the streamline does not have points in the ROI(s) but intersects it,
    new points are generated.

    Parameters
    ----------
    orig_strl: np.ndarray
        The original streamline.
    inter_vox: np.ndarray
        The intersection points of the streamline with the voxel grid.
    in_vox_idx: int
        The index of the voxel where the streamline enters.
    out_vox_idx: int
        The index of the voxel where the streamline exits.
    points_to_indices: np.ndarray
        The indices of the voxels in the voxel grid.

    Returns
    -------
    segment: np.ndarray
        The segment of the streamline that is in the voxel.
    """

    additional_start_pt = None
    additional_exit_pt = None
    nb_add_points = 0

    # Check if the ROI contains a real streamline point at
    # the beginning of the streamline
    in_strl_point = _get_streamline_pt_index(points_to_indices,
                                             in_vox_idx)

    # If not, find the next real streamline point
    if in_strl_point is None:
        # Find the index of the next real streamline point
        in_strl_point = get_next_real_point(points_to_indices, in_vox_idx)
        # Generate an artificial point on the line between the previous
        # real point and the next real point
        additional_start_pt = _get_point_on_line(orig_strl[in_strl_point - 1],
                                                 orig_strl[in_strl_point],
                                                 inter_vox[in_vox_idx])
        nb_add_points += 1

    # Check if the ROI contains a real streamline point at
    # the end of the streamline
    out_strl_point = _get_streamline_pt_index(points_to_indices,
                                              out_vox_idx,
                                              from_start=False)
    # If not, find the previous real streamline point
    if out_strl_point is None:
        # Find the index of the previous real streamline point
        out_strl_point = get_previous_real_point(points_to_indices,
                                                 out_vox_idx)
        # Generate an artificial point on the line between the previous
        # real point and the next real point
        additional_exit_pt = _get_point_on_line(orig_strl[out_strl_point],
                                                orig_strl[out_strl_point + 1],
                                                inter_vox[out_vox_idx])
        nb_add_points += 1

    # Compute the number of points in the cut streamline and
    # add the number of artificial points
    nb_points_orig_strl = out_strl_point - in_strl_point + 1
    nb_points = nb_points_orig_strl + nb_add_points
    orig_segment_len = len(orig_strl[in_strl_point:out_strl_point + 1])

    # TODO: Fix the bug in `uncompress` and remove this
    # There is a bug with `uncompress` where the number of `points_to_indices`
    # is not the same as the number of points in the streamline. This is
    # a temporary fix.
    segment_len = min(
        nb_points,
        orig_segment_len + nb_add_points)
    # Initialize the new streamline segment
    segment = np.zeros((segment_len, 3))
    # offset for indexing in case there are new points
    offset = 0

    # If there is a new point at the beginning of the streamline
    # add it to the segment
    if additional_start_pt is not None:
        segment[0] = additional_start_pt
        offset += 1

    # Set the segment as the part of the original streamline that is
    # in the ROI

    # Note: this works correctly even in the case where the "previous"
    # point is the same or lower than the entry point, because of
    # numpy indexing
    segment[offset:offset + nb_points_orig_strl] = \
        orig_strl[in_strl_point:out_strl_point + 1]

    # If there is a new point at the end of the streamline
    # add it to the segment.
    if additional_exit_pt is not None:
        segment[-1] = additional_exit_pt

    # Return the segment
    return segment
