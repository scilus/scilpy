# -*- coding: utf-8 -*-

import itertools

from dipy.tracking.metrics import length
from dipy.tracking.streamline import (set_number_of_points,
                                      transform_streamlines)
from dipy.tracking._utils import _mapping_to_voxel
from dipy.tracking.vox2track import _streamlines_in_mask
from nibabel.affines import apply_affine
import numpy as np


def target_line_based(streamlines, target_mask, affine=None, include=True):
    # Copy-Paste from Dipy to get indices
    target_mask = np.array(target_mask, dtype=np.uint8, copy=True)
    lin_T, offset = _mapping_to_voxel(affine)

    streamline_index = _streamlines_in_mask(
        streamlines, target_mask, lin_T, offset)

    target_indices = []
    target_streamlines = []
    for idx in np.where(streamline_index == [0, 1][include])[0]:
        target_indices.append(idx)
        target_streamlines.append(streamlines[idx])

    return target_streamlines, target_indices


def filter_grid_roi(sft, mask, filter_type, is_not):
    streamlines = list(sft.streamlines)
    transfo, _, _, _ = sft.space_attributes

    line_based_indices = []
    if filter_type == 'any':
        _, line_based_indices = target_line_based(
            streamlines, mask, transfo)
    else:
        # For endpoint filtering, we need to keep 2 informations
        # Could be faster for either end, but the code look cleaner like this
        inv_transfo = np.linalg.inv(transfo)
        inv_transfo[0:3, 3] += 0.5
        streamline_vox = transform_streamlines(streamlines,
                                               inv_transfo)
        line_based_indices_1 = []
        line_based_indices_2 = []
        for i, line_vox in enumerate(streamline_vox):
            voxel_1 = (int(line_vox[0][0]),
                       int(line_vox[0][1]),
                       int(line_vox[0][2]))
            voxel_2 = (int(line_vox[-1][0]),
                       int(line_vox[-1][1]),
                       int(line_vox[-1][2]))
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

    line_based_indices = np.asarray(line_based_indices)

    # If the --not option is used, the selection is inverted
    all_indices = range(len(streamlines))
    if is_not:
        line_based_indices = np.setdiff1d(all_indices,
                                          np.unique(line_based_indices))

    # From indices to streamlines
    final_streamlines = list(
        sft.streamlines[line_based_indices.astype(np.int32)])

    return final_streamlines, line_based_indices


def pre_filtering_for_geometrical_shape(sft, size,
                                        center, filter_type,
                                        is_in_vox):
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
                     filter_type, is_not, is_in_vox=False):
    pre_filtered_streamlines, pre_filtered_indices = \
        pre_filtering_for_geometrical_shape(sft, ellipsoid_radius,
                                            ellipsoid_center, filter_type,
                                            is_in_vox)

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
    # If the --not option is used, the selection is inverted
    all_indices = range(len(sft))
    if is_not:
        selected_by_ellipsoid = np.setdiff1d(all_indices,
                                             np.unique(selected_by_ellipsoid))

    # From indices to streamlines
    final_streamlines = list(sft.streamlines[
        np.asarray(selected_by_ellipsoid).astype(np.int32)])

    return final_streamlines, selected_by_ellipsoid


def filter_cuboid(sft, cuboid_radius, cuboid_center,
                  filter_type, is_not):

    pre_filtered_streamlines, pre_filtered_indices = \
        pre_filtering_for_geometrical_shape(sft, cuboid_radius,
                                            cuboid_center, filter_type,
                                            False)

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
            nb_points = max(int(length(line)/np.average(res) * 10), 2)
            line = set_number_of_points(line, nb_points)
            points_in_cuboid = np.abs(line - cuboid_center) / cuboid_radius

            points_in_cuboid[points_in_cuboid <= 1] = 1
            points_in_cuboid[points_in_cuboid > 1] = 0
            points_in_cuboid = np.sum(points_in_cuboid, axis=1)

            if np.argwhere(points_in_cuboid == 3).any():
                # If at least one point was in the cuboid, we selected
                # the streamlines
                selected_by_cuboid.append(pre_filtered_indices[i])
        else:
            # Faster to do it twice than trying to do in using an array of 2
            points_in_cuboid = np.abs(line[0] - cuboid_center) / cuboid_radius
            points_in_cuboid[points_in_cuboid <= 1] = 1
            points_in_cuboid[points_in_cuboid > 1] = 0
            points_in_cuboid = np.sum(points_in_cuboid)

            if points_in_cuboid == 3:
                line_based_indices_1.append(pre_filtered_indices[i])

            points_in_cuboid = np.abs(line[-1] - cuboid_center) / cuboid_radius
            points_in_cuboid[points_in_cuboid <= 1] = 1
            points_in_cuboid[points_in_cuboid > 1] = 0
            points_in_cuboid = np.sum(points_in_cuboid)

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

    # If the --not option is used, the selection is inverted
    all_indices = range(len(sft))
    if is_not:
        selected_by_cuboid = np.setdiff1d(all_indices,
                                          np.unique(selected_by_cuboid))

    # From indices to streamlines
    # From indices to streamlines
    final_streamlines = list(sft.streamlines[
        np.asarray(selected_by_cuboid).astype(np.int32)])

    return final_streamlines, selected_by_cuboid
