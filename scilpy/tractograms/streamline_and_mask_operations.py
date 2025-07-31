# -*- coding: utf-8 -*-
from enum import Enum
from multiprocessing import Pool

import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin

from nibabel.streamlines import ArraySequence

from scipy.ndimage import map_coordinates

from scilpy.tractograms.uncompress import streamlines_to_voxel_coordinates
from scilpy.tractograms.streamline_operations import \
    (_get_point_on_line, _get_streamline_pt_index,
     _get_next_real_point, _get_previous_real_point,
     filter_streamlines_by_length,
     resample_streamlines_step_size)


class CuttingStyle(Enum):
    DEFAULT = 0,
    KEEP_LONGEST = 1
    TRIM_ENDPOINTS = 2


def get_endpoints_density_map(sft, point_to_select=1, to_millimeters=False):
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
    to_millimeters: bool
        Resample the streamlines to have a step size of 1 mm. This
        allows the user to compute endpoints with mms instead of points.
        Especially useful with compressed streamlines.

    Returns
    -------
    np.ndarray: A np.ndarray where voxel values represent the density of
        endpoints.
    """

    endpoints_map_head, endpoints_map_tail = \
        get_head_tail_density_maps(sft, point_to_select, to_millimeters)
    return endpoints_map_head + endpoints_map_tail


def get_head_tail_density_maps(sft, point_to_select=1, to_millimeters=False):
    """
    Compute two separate endpoints density maps for the head and tail of
    a list of streamlines.

    Parameters
    ----------
    sft: StatefulTractogram
        The streamlines to compute endpoints density from.
    point_to_select: int
        Instead of computing the density based on the first and last points,
        select more than one at each end.
    to_millimeters: bool
        Resample the streamlines to have a step size of 1 mm. This
        allows the user to compute endpoints with mms instead of points.
        Especially useful with compressed streamlines.

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

    if to_millimeters:
        # Resample the streamlines to have a step size of 1 mm
        streamlines = resample_streamlines_step_size(sft, 1.0).streamlines
    else:
        streamlines = sft.streamlines

    dimensions = sft.dimensions
    # Get the indices of the voxels intersected
    list_indices, points_to_indices = streamlines_to_voxel_coordinates(
        streamlines, return_mapping=True)

    # Initialize the endpoints maps
    endpoints_map_head = np.zeros(dimensions)
    endpoints_map_tail = np.zeros(dimensions)

    # A possible optimization would be to compute all coordinates first
    # and then do the np.add.at only once.
    for indices, points in zip(list_indices, points_to_indices):

        # Get the head and tail coordinates
        # +1 to include the last point
        point_to_select = min(point_to_select, len(points) - 1)
        head_indices = indices[:points[point_to_select] + 1, :]
        tail_indices = indices[points[-point_to_select]:, :]

        # Add the points to the endpoints map
        # Note: np.add.at is used to support duplicate points
        np.add.at(endpoints_map_head, tuple(head_indices.T), 1)
        np.add.at(endpoints_map_tail, tuple(tail_indices.T), 1)

    return endpoints_map_head, endpoints_map_tail


def _trim_streamline_in_mask(
    idx, streamline, pts_to_idx, mask
):
    """ Trim streamlines to the bounding box or a binary mask. More
    streamlines may be generated if the original streamline goes in and out
    of the mask.

    Parameters
    ----------
    idx: np.ndarray
        Indices of the voxels intersected by the streamline.
    streamline: np.ndarray
        The streamlines to cut.
    pts_to_idx: np.ndarray
        Mapping from streamline points to indices.
    mask: np.ndarray
        Boolean array representing the region.

    Returns
    -------
    new_strmls : list of np.ndarray
        New streamlines trimmed within the mask.
    """

    # Find all the points of the streamline that are in the ROIs
    roi_data_1_intersect = map_coordinates(
        mask, idx.T, order=0, mode='constant', cval=0)

    # Select the points that are not in the mask
    split_idx = np.arange(len(roi_data_1_intersect))[
        roi_data_1_intersect == 0]
    # Split the streamline into segments that are in the mask
    split_strmls = np.array_split(np.arange(len(roi_data_1_intersect)),
                                  split_idx)
    new_strmls = []
    for strml in split_strmls:
        if len(strml) <= 3:
            continue
        # Get the entry and exit points for each segment
        # Skip the first point as it caused the split
        in_strl_idx, out_strl_idx = strml[1], strml[-1]
        cut_strl = compute_streamline_segment(streamline, idx,
                                              in_strl_idx, out_strl_idx,
                                              pts_to_idx)
        new_strmls.append(cut_strl)

    return new_strmls


def _trim_streamline_endpoints_in_mask(
    idx, streamline, pts_to_idx, mask
):
    """ Trim a streamline to remove its endpoints if they are outside of
    a mask. This function does not generate new streamlines.

    Parameters
    ----------
    idx: np.ndarray
        Indices of the voxels intersected by the streamline.
    streamline: np.ndarray
        The streamlines to cut.
    pts_to_idx: np.ndarray
        Mapping from streamline points to indices.
    mask: np.ndarray
        Boolean array representing the region.

    Returns
    -------
    streamline: np.ndarray
        The trimmed streamline within the mask.
    """

    # Find all the points of the streamline that are in the ROIs
    roi_data_1_intersect = map_coordinates(
        mask, idx.T, order=0, mode='constant', cval=0)

    # Select the points that are in the mask
    mask_idx = np.arange(len(roi_data_1_intersect))[
        roi_data_1_intersect == 1]

    if len(mask_idx) == 0:
        return []

    # Get the entry and exit points for each segment
    in_strl_idx = np.amin(mask_idx)
    out_strl_idx = np.amax(mask_idx)
    cut_strl = compute_streamline_segment(streamline, idx,
                                          in_strl_idx, out_strl_idx,
                                          pts_to_idx)
    return [cut_strl]


def _trim_streamline_in_mask_keep_longest(
    idx, streamline, pts_to_idx, mask
):
    """ Trim a streamline to keep the longest segment within a mask. This
    function does not generate new streamlines.

    Parameters
    ----------
    idx: np.ndarray
        Indices of the voxels intersected by the streamline.
    streamline: np.ndarray
        The streamlines to cut.
    pts_to_idx: np.ndarray
        Mapping from streamline points to indices.
    mask: np.ndarray
        Boolean array representing the region.

    Returns
    -------
    streamline: np.ndarray
        The trimmed streamline within the mask.
    """
    # Find all the points of the streamline that are in the ROIs
    roi_data_1_intersect = map_coordinates(
        mask, idx.T, order=0, mode='constant', cval=0)

    # Select the points that are not in the mask
    split_idx = np.arange(len(roi_data_1_intersect))[
        roi_data_1_intersect == 0]
    # Split the streamline into segments that are in the mask
    split_strmls = np.array_split(np.arange(len(roi_data_1_intersect)),
                                  split_idx)

    # Find the longest segment of the streamline that is in the mask
    longest_strml = max(split_strmls, key=len)

    if len(longest_strml) <= 1:
        return []
    # Get the entry and exit points for the longest segment
    # Skip the first point as it caused the split
    id_to_pick = 0 if np.count_nonzero(roi_data_1_intersect) == len(idx) else 1
    in_strl_idx, out_strl_idx = longest_strml[id_to_pick], longest_strml[-1]
    cut_strl = compute_streamline_segment(streamline, idx,
                                          in_strl_idx, out_strl_idx,
                                          pts_to_idx)
    return [cut_strl]


def cut_streamlines_with_mask(sft, mask,
                              cutting_style=CuttingStyle.DEFAULT,
                              min_len=0, processes=1):
    """
    Cut streamlines according to a binary mask. This function erases the
    data_per_point.

    If keep_longest is set, the longest segment of the streamline that crosses
    the mask will be kept. Otherwise, the streamline will be cut at the mask.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to cut streamlines (using a single mask with 1 entity) from.
    mask: np.ndarray
        Boolean array representing the region (must contain 1 entity)
    cutting_style: CuttingStyle
        How to cut the streamlines. Default is to cut the streamlines at the
        mask. If keep_longest is set, the longest segment of the streamline
        that crosses the mask will be kept. If trim_endpoints is set, the
        endpoints of the streamlines will be cut but the middle part of the
        streamline may go outside the mask.
    min_len: float
        Minimum length from the resulting streamlines.
    processes: int
        Number of processes to use.

    Returns
    -------
    new_sft : StatefulTractogram
        New object with the streamlines trimmed within the mask.
    """

    orig_space = sft.space
    orig_origin = sft.origin
    sft.to_vox()
    sft.to_corner()

    # Get the indices of the voxels
    # intersected by the streamlines and the mapping from points to indices
    indices, points_to_idx = streamlines_to_voxel_coordinates(
        sft.streamlines,
        return_mapping=True
    )

    if len(sft.streamlines[0]) != len(points_to_idx[0]):
        raise ValueError("Error in the streamlines_to_voxel_coordinates "
                         "function. Try running the "
                         "scil_tractogram_remove_invalid.py script with the \n"
                         "--remove_single_point and "
                         "--remove_overlapping_points options.")

    # Select the trimming function. If keep_longest is set, the longest
    # segment of the streamline that crosses the mask will be kept. If
    # trim_endpoints is set, the endpoints of the streamlines will be cut.
    # Otherwise, the streamline will be cut at the mask.
    if cutting_style == CuttingStyle.TRIM_ENDPOINTS:
        trim_func = _trim_streamline_endpoints_in_mask
    elif cutting_style == CuttingStyle.KEEP_LONGEST:
        trim_func = _trim_streamline_in_mask_keep_longest
    else:
        trim_func = _trim_streamline_in_mask

    # Trim streamlines with the mask and return the new streamlines
    pool = Pool(processes)
    lists_of_new_strmls = pool.starmap(
        trim_func, [(i, s, pt, mask) for (i, s, pt) in zip(
            indices, sft.streamlines, points_to_idx)])
    pool.close()
    # Flatten the list of lists of new streamlines in a single list of
    # new streamlines
    new_strmls = ArraySequence([strml for list_of_strml in lists_of_new_strmls
                                for strml in list_of_strml])

    new_sft = StatefulTractogram.from_sft(
        new_strmls, sft)
    
    # Put back the original space and origin
    new_sft.to_space(orig_space)
    new_sft.to_origin(orig_origin)
    sft.to_space(orig_space)
    sft.to_origin(orig_origin)

    new_sft, *_ = filter_streamlines_by_length(new_sft, min_length=min_len)

    return new_sft


def cut_streamlines_between_labels(
    sft, label_data, label_ids=None, min_len=0,
    one_point_in_roi=False, no_point_in_roi=False, processes=1
):
    """
    Cut streamlines so their segment are going from blob #1 to blob #2 in a
    binary mask. This function presumes strictly two blobs are present in the
    mask.

    This function erases the data_per_point.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to cut streamlines (using a single mask with 2 entities) from.
    label_data: np.ndarray
        Label map representing the two regions.
    label_ids: list of int, optional
        The two labels to cut between. If not provided, the two unique labels
        in the label map will be used.
    min_len: float
        Minimum length from the resulting streamlines.
    one_point_in_roi: bool
        If True, one point in each ROI will be kept.
    no_point_in_roi: bool
        If True, no point in the ROIs will be kept.

    Returns
    -------
    new_sft : StatefulTractogram
        New object with the streamlines trimmed within the masks.
    """
    orig_space = sft.space
    orig_origin = sft.origin
    sft.to_vox()
    sft.to_corner()
    if label_ids is None:
        unique_vals = np.unique(label_data[label_data != 0])
        if len(unique_vals) != 2:
            raise ValueError('More than two values in the label file, '
                             'please select specific label ids.')
    else:
        unique_vals = label_ids

    # Create two binary masks
    label_data_1 = np.copy(label_data)
    mask = label_data_1 != unique_vals[0]
    label_data_1[mask] = 0

    label_data_2 = np.copy(label_data)
    mask = label_data_2 != unique_vals[1]
    label_data_2[mask] = 0

    (indices, points_to_idx) = streamlines_to_voxel_coordinates(
        sft.streamlines,
        return_mapping=True
    )

    if len(sft.streamlines[0]) != len(points_to_idx[0]):
        raise ValueError("Error in the streamlines_to_voxel_coordinates "
                         "function. Try running the "
                         "scil_tractogram_remove_invalid.py script with the \n"
                         "--remove_single_point and "
                         "--remove_overlapping_points options.")

    # Trim streamlines with the mask and return the new streamlines
    pool = Pool(processes)
    lists_of_new_strmls = pool.starmap(
        _cut_streamline_with_labels, [(i, s, pt, label_data_1, label_data_2,
                                       one_point_in_roi, no_point_in_roi)
                                      for (i, s, pt) in zip(
                                          indices, sft.streamlines,
                                          points_to_idx)])
    pool.close()
    # Flatten the list of lists of new streamlines in a single list of
    # new streamlines
    list_of_new_strmls = [strml for strml in lists_of_new_strmls
                          if strml is not None]
    new_strmls = ArraySequence(list_of_new_strmls)

    new_sft = StatefulTractogram.from_sft(new_strmls, sft)

    # Put back the original space and origin
    new_sft.to_space(orig_space)
    new_sft.to_origin(orig_origin)
    sft.to_space(orig_space)
    sft.to_origin(orig_origin)

    new_sft, *_ = filter_streamlines_by_length(new_sft, min_length=min_len)

    return new_sft


def _cut_streamline_with_labels(
    idx, streamline, pts_to_idx, roi_data_1, roi_data_2,
    one_point_in_roi=False, no_point_in_roi=False
):
    """
    Cut streamlines so their segment are going from label mask #1 to label
    mask #2. New endpoints may be generated to maximize the streamline length
    within the masks.

    Parameters
    ----------
    idx: np.ndarray
        Indices of the voxels intersected by the streamlines.
    streamline: np.ndarray
        The streamlines to cut.
    pts_to_idx: np.ndarray
        Mapping from points to indices.
    roi_data_1: np.ndarray
        Boolean array representing the region #1.
    roi_data_2: np.ndarray
        Boolean array representing the region #2.
    one_point_in_roi: bool
        If True, one point in each ROI will be kept.
    no_point_in_roi: bool
        If True, no point in the ROIs will be kept.

    Returns
    -------
    new_strmls : list of np.ndarray
        New streamlines trimmed within the masks.
    """
    # Find the first and last "voxels" of the streamline that are in the
    # ROIs
    in_strl_idx, out_strl_idx = _intersects_two_rois(
        roi_data_1, roi_data_2, idx, one_point_in_roi=one_point_in_roi,
        no_point_in_roi=no_point_in_roi)
    cut_strl = None
    # If the streamline intersects both ROIs
    if in_strl_idx is not None and out_strl_idx is not None:
        # Compute the new streamline by keeping only the segment between
        # the two ROIs
        cut_strl = compute_streamline_segment(streamline, idx,
                                              in_strl_idx, out_strl_idx,
                                              pts_to_idx)
    return cut_strl


def _get_all_streamline_segments_in_roi(all_strl_indices):
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

    split_pos = np.where(strl_indices_grad != 1)[0] + 1

    # Covers weird cases where there is only non consecutive indices
    if len(strl_indices_grad) == len(split_pos) + 1:
        return [None]

    # Split the indices of the voxels intersecting with the ROIs into
    # segments where the gradient is 1 (i.e a chunk of consecutive indices)
    strl_indices_split = np.split(all_strl_indices, split_pos)

    return [sublist for sublist in strl_indices_split if sublist.size > 0]


def _get_in_and_out_strl_indices(in_strl_indices_split, out_strl_indices_split,
                                 one_point_in_roi=False,
                                 no_point_in_roi=False):
    """
    Get the first and last "voxels" of the streamline

    Parameters
    ----------
    in_strl_indices_split: list
        List of np.array of streamline segment indices (N)
    out_strl_indices_split: list
        List of np.array of streamline segment indices (N)
    one_point_in_roi: bool
        If True, one point in each ROI will be kept.
    no_point_in_roi: bool
        If True, no point in the ROIs will be kept.

    Returns
    -------
    in_strl_idx : int
        index of the first point of the streamline
    out_strl_idx : int
        index of the last point of the streamline
    """

    # One of them is None takes the first segment of the other
    if in_strl_indices_split[0] is None:
        return None, out_strl_indices_split[0][-1]
    elif out_strl_indices_split[0] is None:
        return in_strl_indices_split[-1][0], None
    else:
        # Check the order of the first segments
        if min(in_strl_indices_split[0]) > min(out_strl_indices_split[0]):
            in_strl_indices_split, out_strl_indices_split = \
                out_strl_indices_split, in_strl_indices_split

        # Get the last segment in the first ROI
        # Get the first segment in the second ROI
        in_strl_indices = in_strl_indices_split[-1]
        out_strl_indices = out_strl_indices_split[0]

    # If no options are set, start the streamline with the first
    # and last point of each segment
    if not one_point_in_roi and not no_point_in_roi:
        in_strl_idx = in_strl_indices[0]
        out_strl_idx = out_strl_indices[-1]
    else:
        if one_point_in_roi:
            add_indice = 0
        elif no_point_in_roi:
            add_indice = 1

        in_strl_idx = in_strl_indices[-1] + add_indice
        out_strl_idx = out_strl_indices[0] - add_indice

    return in_strl_idx, out_strl_idx


def _intersects_two_rois(roi_data_1, roi_data_2, strl_indices,
                         one_point_in_roi=False, no_point_in_roi=False):
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
    one_point_in_roi: bool
        If True, one point in each ROI will be kept.
    no_point_in_roi: bool
        If True, no point in the ROIs will be kept.

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
        in_strl_indices = _get_all_streamline_segments_in_roi(
            in_strl_indices)

    if len(out_strl_indices) == 0:
        out_strl_indices = [None]
    else:
        out_strl_indices = _get_all_streamline_segments_in_roi(
            out_strl_indices)

    if in_strl_indices[0] is None and out_strl_indices[0] is None:
        return None, None
    else:
        in_strl_idx, out_strl_idx = _get_in_and_out_strl_indices(
            in_strl_indices,
            out_strl_indices,
            one_point_in_roi,
            no_point_in_roi)

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

    # Check if the ROI contains a real streamline point at
    # the beginning of the streamline
    in_strl_point = _get_streamline_pt_index(points_to_indices,
                                             in_vox_idx)

    # If not, find the next real streamline point
    if in_strl_point is None:
        # Find the index of the next real streamline point
        in_strl_point = _get_next_real_point(points_to_indices, in_vox_idx)

        if in_strl_point == 0:
            # If the entry point is the first point of the streamline,
            # don't generate a new point
            additional_start_pt = None
        else:

            # Generate an artificial point on the line between the previous
            # real point and the next real point
            additional_start_pt = _get_point_on_line(
                orig_strl[in_strl_point - 1], orig_strl[in_strl_point],
                inter_vox[in_vox_idx])
    # Check if the ROI contains a real streamline point at
    # the end of the streamline
    out_strl_point = _get_streamline_pt_index(points_to_indices,
                                              out_vox_idx,
                                              from_start=False)
    # If not, find the previous real streamline point
    if out_strl_point is None:
        # Find the index of the previous real streamline point
        out_strl_point = _get_previous_real_point(points_to_indices,
                                                  out_vox_idx)

        if out_strl_point == len(points_to_indices) - 1:
            # If the exit point is the last point of the streamline,
            # don't generate a new point
            additional_exit_pt = None
        else:
            # Generate an artificial point on the line between the previous
            # real point and the next real point
            additional_exit_pt = _get_point_on_line(
                orig_strl[out_strl_point], orig_strl[out_strl_point + 1],
                inter_vox[out_vox_idx])

    # Set the segment as the part of the original streamline that is
    # in the ROI
    segment = orig_strl[in_strl_point:out_strl_point + 1]

    # Whereas the original implementation was using offsets to include
    # additional points, we are now inserting and appending to simplify
    # the code and avoid introducing more bugs.

    # If there is a new point at the beginning of the streamline
    # add it to the segment
    if additional_start_pt is not None:
        segment = np.insert(segment, 0, [additional_start_pt], axis=0)

    # If there is a new point at the end of the streamline
    # add it to the segment.
    if additional_exit_pt is not None:
        segment = np.append(segment, [additional_exit_pt], axis=0)

    # Return the segment
    return segment
