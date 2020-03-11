# -*- coding: utf-8 -*-

import itertools

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.metrics import length
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.vox2track import _streamlines_in_mask
from nibabel.affines import apply_affine
import numpy as np


def streamlines_in_mask(sft, target_mask):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    target_mask : numpy.ndarray
        Binary mask in which the streamlines should pass.
    Returns
    -------
    ids : list
        Ids of the streamlines passing through the mask.
    """
    sft.to_vox()
    sft.to_corner()
    # Copy-Paste from Dipy to get indices
    target_mask = np.array(target_mask, dtype=np.uint8, copy=True)

    streamlines_case = _streamlines_in_mask(list(sft.streamlines),
                                            target_mask, np.eye(3),
                                            [-0.5, -0.5, -0.5])

    return np.where(streamlines_case == [0, 1][True])[0].tolist()


def filter_grid_roi(sft, mask, filter_type, is_exclude):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    target_mask : numpy.ndarray
        Binary mask in which the streamlines should pass.
    filter_type: str
        One of the 3 following choices, 'any', 'either_end', 'both_ends'.
    is_exclude: bool
        Value to indicate if the ROI is an AND (false) or a NOT (true).
    Returns
    -------
    ids : tuple
        Filtered sft.
        Ids of the streamlines passing through the mask.
    """
    line_based_indices = []
    if filter_type == 'any':
        line_based_indices = streamlines_in_mask(sft, mask)
    else:
        sft.to_vox()
        sft.to_corner()
        streamline_vox = sft.streamlines
        # For endpoint filtering, we need to keep 2 separately
        # Could be faster for either end, but the code look cleaner like this
        line_based_indices_1 = []
        line_based_indices_2 = []
        for i, line_vox in enumerate(streamline_vox):
            voxel_1 = tuple(line_vox[0].astype(np.int16))
            voxel_2 = tuple(line_vox[-1].astype(np.int16))
            if mask[voxel_1]:
                line_based_indices_1.append(i)
            if mask[voxel_2]:
                line_based_indices_2.append(i)

        # Both endpoints need to be in the mask (AND)
        if filter_type == 'both_ends':
            line_based_indices = np.intersect1d(line_based_indices_1,
                                                line_based_indices_2)
        # Only one endpoint need to be in the mask (OR)
        elif filter_type == 'either_end':
            line_based_indices = np.union1d(line_based_indices_1,
                                            line_based_indices_2)

    # If the 'exclude' option is used, the selection is inverted
    if is_exclude:
        line_based_indices = np.setdiff1d(range(len(sft)),
                                          np.unique(line_based_indices))
    line_based_indices = np.asarray(line_based_indices).astype(np.int32)

    # From indices to sft
    streamlines = sft.streamlines[line_based_indices]
    data_per_streamline = sft.data_per_streamline[line_based_indices]
    data_per_point = sft.data_per_point[line_based_indices]

    new_sft = StatefulTractogram.from_sft(streamlines, sft,
                                          data_per_streamline=data_per_streamline,
                                          data_per_point=data_per_point)

    return new_sft, line_based_indices


def pre_filtering_for_geometrical_shape(sft, size,
                                        center, filter_type,
                                        is_in_vox):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    size : numpy.ndarray (3)
        Size in mm, x/y/z of the ROI.
    center: numpy.ndarray (3)
        Center x/y/z of the ROI.
    filter_type: str
        One of the 3 following choices, 'any', 'either_end', 'both_ends'.
    is_in_vox: bool
        Value to indicate if the ROI is in voxel space.
    Returns
    -------
    ids : tuple
        Filtered sft.
        Ids of the streamlines passing through the mask.
    """
    transfo, dim, _, _ = sft.space_attributes
    inv_transfo = np.linalg.inv(transfo)

    # Create relevant info about the ellipsoid in vox/world space
    if is_in_vox:
        center = np.asarray(apply_affine(transfo, center))
    bottom_corner = center - size
    top_corner = center + size
    x_val = [bottom_corner[0], top_corner[0]]
    y_val = [bottom_corner[1], top_corner[1]]
    z_val = [bottom_corner[2], top_corner[2]]
    corner_world = list(itertools.product(x_val, y_val, z_val))
    corner_vox = apply_affine(inv_transfo, corner_world)

    # Since the filtering using a grid is so fast, we pre-filter
    # using a BB around the ellipsoid
    min_corner = np.min(corner_vox, axis=0) - 1.0
    max_corner = np.max(corner_vox, axis=0) + 1.5
    pre_mask = np.zeros(dim)
    min_x, max_x = int(max(min_corner[0], 0)), int(min(max_corner[0], dim[0]))
    min_y, max_y = int(max(min_corner[1], 0)), int(min(max_corner[1], dim[1]))
    min_z, max_z = int(max(min_corner[2], 0)), int(min(max_corner[2], dim[2]))

    pre_mask[min_x:max_x, min_y:max_y, min_z:max_z] = 1

    return filter_grid_roi(sft, pre_mask, filter_type, False)


def filter_ellipsoid(sft, ellipsoid_radius, ellipsoid_center,
                     filter_type, is_exclude, is_in_vox=False):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    ellipsoid_radius : numpy.ndarray (3)
        Size in mm, x/y/z of the ellipsoid.
    ellipsoid_center: numpy.ndarray (3)
        Center x/y/z of the ellipsoid.
    filter_type: str
        One of the 3 following choices, 'any', 'either_end', 'both_ends'.
    is_exclude: bool
        Value to indicate if the ROI is an AND (false) or a NOT (true).
    is_in_vox: bool
        Value to indicate if the ROI is in voxel space.
    Returns
    -------
    ids : tuple
        Filtered sft.
        Ids of the streamlines passing through the mask.
    """
    pre_filtered_sft, pre_filtered_indices = \
        pre_filtering_for_geometrical_shape(sft, ellipsoid_radius,
                                            ellipsoid_center, filter_type,
                                            is_in_vox)
    pre_filtered_sft.to_rasmm()
    pre_filtered_sft.to_center()
    pre_filtered_streamlines = pre_filtered_sft.streamlines
    transfo, _, res, _ = sft.space_attributes

    if is_in_vox:
        ellipsoid_center = np.asarray(apply_affine(transfo,
                                                   ellipsoid_center))
    selected_by_ellipsoid = []
    line_based_indices_1 = []
    line_based_indices_2 = []
    # This is still point based (but resampled), I had a ton of problems trying
    # to use something with intersection, but even if I could do it :
    # The result won't be identical to MI-Brain since I am not using the
    # vtkPolydata. Also it won't be identical to TrackVis either,
    # because TrackVis is point-based for Spherical ROI...
    ellipsoid_radius = np.asarray(ellipsoid_radius)
    ellipsoid_center = np.asarray(ellipsoid_center)

    for i, line in enumerate(pre_filtered_streamlines):
        if filter_type == 'any':
            # Resample to 1/10 of the voxel size
            nb_points = max(int(length(line) / np.average(res) * 10), 2)
            line = set_number_of_points(line, nb_points)
            points_in_ellipsoid = np.sum(
                ((line - ellipsoid_center) / ellipsoid_radius) ** 2,
                axis=1)
            if np.argwhere(points_in_ellipsoid <= 1).any():
                # If at least one point was in the ellipsoid, we selected
                # the streamline
                selected_by_ellipsoid.append(pre_filtered_indices[i])
        else:
            points_in_ellipsoid = np.sum(
                ((line[0] - ellipsoid_center) / ellipsoid_radius) ** 2)

            if points_in_ellipsoid <= 1.0:
                line_based_indices_1.append(pre_filtered_indices[i])

            points_in_ellipsoid = np.sum(
                ((line[-1] - ellipsoid_center) / ellipsoid_radius) ** 2)
            if points_in_ellipsoid <= 1.0:
                line_based_indices_2.append(pre_filtered_indices[i])

    # Both endpoints need to be in the mask (AND)
    if filter_type == 'both_ends':
        selected_by_ellipsoid = np.intersect1d(line_based_indices_1,
                                               line_based_indices_2)
    # Only one endpoint needs to be in the mask (OR)
    elif filter_type == 'either_end':
        selected_by_ellipsoid = np.union1d(line_based_indices_1,
                                           line_based_indices_2)

    # If the 'exclude' option is used, the selection is inverted
    if is_exclude:
        selected_by_ellipsoid = np.setdiff1d(range(len(sft)),
                                             np.unique(selected_by_ellipsoid))
    line_based_indices = np.asarray(selected_by_ellipsoid).astype(np.int32)

    # From indices to sft
    streamlines = sft.streamlines[line_based_indices]
    data_per_streamline = sft.data_per_streamline[line_based_indices]
    data_per_point = sft.data_per_point[line_based_indices]

    new_sft = StatefulTractogram.from_sft(streamlines, sft,
                                          data_per_streamline=data_per_streamline,
                                          data_per_point=data_per_point)

    return new_sft, line_based_indices


def filter_cuboid(sft, cuboid_radius, cuboid_center,
                  filter_type, is_exclude):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    cuboid_radius : numpy.ndarray (3)
        Size in mm, x/y/z of the cuboid.
    cuboid_center: numpy.ndarray (3)
        Center x/y/z of the cuboid.
    filter_type: str
        One of the 3 following choices, 'any', 'either_end', 'both_ends'.
    is_exclude: bool
        Value to indicate if the ROI is an AND (false) or a NOT (true).
    is_in_vox: bool
        Value to indicate if the ROI is in voxel space.
    Returns
    -------
    ids : tuple
        Filtered sft.
        Ids of the streamlines passing through the mask.
    """
    pre_filtered_sft, pre_filtered_indices = \
        pre_filtering_for_geometrical_shape(sft, cuboid_radius,
                                            cuboid_center, filter_type,
                                            False)
    pre_filtered_sft.to_rasmm()
    pre_filtered_sft.to_center()
    pre_filtered_streamlines = pre_filtered_sft.streamlines
    _, _, res, _ = sft.space_attributes

    selected_by_cuboid = []
    line_based_indices_1 = []
    line_based_indices_2 = []
    # Also here I am not using a mathematical intersection and
    # I am not using vtkPolyData like in MI-Brain, so not exactly the same
    cuboid_radius = np.asarray(cuboid_radius)
    cuboid_center = np.asarray(cuboid_center)
    for i, line in enumerate(pre_filtered_streamlines):
        if filter_type == 'any':
            # Resample to 1/10 of the voxel size
            nb_points = max(int(length(line) / np.average(res) * 10), 2)
            line = set_number_of_points(line, nb_points)
            points_in_cuboid = np.abs(line - cuboid_center) / cuboid_radius
            points_in_cuboid = np.sum(np.where(points_in_cuboid <= 1, 1, 0),
                                      axis=1)

            if np.argwhere(points_in_cuboid == 3).any():
                # If at least one point was in the cuboid in x/y/z,
                # we selected that streamline
                selected_by_cuboid.append(pre_filtered_indices[i])
        else:
            # Faster to do it twice than trying to do in using an array of 2
            points_in_cuboid = np.abs(line[0] - cuboid_center) / cuboid_radius
            points_in_cuboid = np.sum(np.where(points_in_cuboid <= 1, 1, 0))

            if points_in_cuboid == 3:
                line_based_indices_1.append(pre_filtered_indices[i])

            points_in_cuboid = np.abs(line[-1] - cuboid_center) / cuboid_radius
            points_in_cuboid = np.sum(np.where(points_in_cuboid <= 1, 1, 0))

            if points_in_cuboid == 3:
                line_based_indices_2.append(pre_filtered_indices[i])

    # Both endpoints need to be in the mask (AND)
    if filter_type == 'both_ends':
        selected_by_cuboid = np.intersect1d(line_based_indices_1,
                                            line_based_indices_2)
    # Only one endpoint need to be in the mask (OR)
    elif filter_type == 'either_end':
        selected_by_cuboid = np.union1d(line_based_indices_1,
                                        line_based_indices_2)

    # If the 'exclude' option is used, the selection is inverted
    if is_exclude:
        selected_by_cuboid = np.setdiff1d(range(len(sft)),
                                          np.unique(selected_by_cuboid))
    line_based_indices = np.asarray(selected_by_cuboid).astype(np.int32)

    # From indices to sft
    streamlines = sft.streamlines[line_based_indices]
    data_per_streamline = sft.data_per_streamline[line_based_indices]
    data_per_point = sft.data_per_point[line_based_indices]

    new_sft = StatefulTractogram.from_sft(streamlines, sft,
                                          data_per_streamline=data_per_streamline,
                                          data_per_point=data_per_point)

    return new_sft, line_based_indices
