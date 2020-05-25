#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enforce a planar-like shape for streamlines. Mainly useful for the corpus
callosum. When streamlines are expected to stay within the same plane as oppose
to abruptly turn. Will facilitate investiguation of commisural heterotopy.

Optionally enforce the left-right symmetry to investiguate commisural homotopy.
Use at your own risk on other bundles.
"""

import argparse

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamline import set_number_of_points
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_reference_arg,
                             add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Path of the input bundle file.')
    p.add_argument('projection_distance', type=float,
                   help='Allowed threshold (mm) for planar projection validity.')
    p.add_argument('inliers',
                   help='Path of the output inliers bundle file.')

    p.add_argument('--outliers',
                   help='Path of the output outliers bundle file.')
    p.add_argument('--symmetry_distance', type=float,
                   help='Enforce symmetry over the X axis (commissural).')
    add_reference_arg(p)
    add_overwrite_arg(p)
    return p


def get_shift_vector(sft):
    dims = sft.space_attributes[1]
    shift_vector = -1.0 * (np.array(dims) / 2.0)

    return shift_vector


def flip_points_in_x(sft, points):
    flip_vector = np.ones(3)
    flip_vector[0] = -1.0
    shift_vector = get_shift_vector(sft)

    flipped_points = []
    for point in points:
        mod_point = point + shift_vector
        mod_point *= flip_vector
        mod_point -= shift_vector
        flipped_points.append(mod_point)

    return flipped_points


def dist_to_plane(normal, plane_point, points):
    # Normal must always be normalized
    normal /= np.linalg.norm(normal)

    # To support entire streamline
    if points.ndim == 2:
        dists = []
        for point in points:
            dist = np.dot(np.array(normal),
                          np.array(point) - np.array(plane_point))
            dists.append(np.abs(dist))
        return np.array(dists)
    # Or a single point
    else:
        dist = np.dot(np.array(normal),
                      np.array(points) - np.array(plane_point))
        return np.abs(dist)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundle)
    assert_outputs_exist(parser, args, args.inliers, args.outliers)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

    ind_inliers = []
    ind_outliers = []

    # Each streamlines is processed individually)
    for i, streamline in enumerate(sft.get_streamlines_copy()):
        points = set_number_of_points(streamline, 10)

        # Use the first, last and midpoint to create a plane
        plane_normal = np.cross(points[0] - points[4],
                                points[0] - points[-1])
        plane_point = points[4]
        distances = dist_to_plane(plane_normal, plane_point, points)

        # Only if the planar test is passed, the symmetry test will follow
        if (distances < args.projection_distance).all():
            if args.symmetry_distance:
                # Simply flip the first point (in X) and check if the last point
                # fall close enough to its flipped counterpart
                sym_first = flip_points_in_x(sft, points)[0]
                if np.linalg.norm(sym_first - points[-1]) < args.symmetry_distance:
                    ind_inliers.append(i)
                else:
                    ind_outliers.append(i)
            else:
                ind_inliers.append(i)
        else:
            ind_outliers.append(i)

    inliers_sft = StatefulTractogram.from_sft(sft.streamlines[ind_inliers], sft,
                                              data_per_point=sft.data_per_point[ind_inliers],
                                              data_per_streamline=sft.data_per_streamline[ind_inliers])
    save_tractogram(inliers_sft, args.inliers)

    if args.outliers:
        outliers_sft = StatefulTractogram.from_sft(sft.streamlines[ind_outliers], sft,
                                                   data_per_point=sft.data_per_point[ind_outliers],
                                                   data_per_streamline=sft.data_per_streamline[ind_outliers])
        save_tractogram(outliers_sft, args.outliers)


if __name__ == "__main__":
    main()
