#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Computes several metrics from an input centroid tractogram and a text file
containing the diameters of each fiber.

Computed metrics:
    - min_external_distance
        Smallest distance separating two fibers.
    - max_voxel_anisotropic
        Diagonal vector of the largest possible anisotropic voxel that would
        not intersect two fibers. (Aligned with streamline ref)
    - max_voxel_isotropic
        Diagonal vector of the largest possible isotropic voxel that would not
        intersect two fibers. (Aligned with streamline ref)
    - true_max_voxel
        Diagonal vector of the largest possible isotropic voxel that
        would not intersect two streamlines. (Rotated from streamline ref)
    - tmv_rotation [optional]
        4D transformation matrix representing the rotation to be applied on
        in_centroids for the alignment of true_max_voxel with the coordinate
        system. (see scil_tractogram_apply_transform.py)
"""
import os
import argparse
import logging
import numpy as np
import nibabel as nib

from scilpy.tractanalysis.fibertube_scoring import (
                                        min_external_distance,
                                        max_voxels,
                                        true_max_voxel)
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg,
                             add_bbox_arg,
                             save_dictionary)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_centroids',
                   help='Path to the tractogram file containing the \n'
                   'fibertubes\' centroids (must be .trk or .tck).')

    p.add_argument('in_diameters',
                   help='Path to a text file containing a list of the \n'
                   'diameters of each fibertube in mm (.txt). Each line \n'
                   'corresponds to the identically numbered centroid.')

    p.add_argument('out_metrics',
                   help='Output file containing the computed metrics \n'
                   '(must be .txt).')

    p.add_argument('--single_diameter', action='store_true',
                   help='If set, the first diameter found in \n'
                   '[in_diameters] will be repeated for each fiber.')

    p.add_argument('--save_tmv_rotation', action='store_true',
                   help='If set, a separate text file will be saved, \n'
                   'containing the transformation matrix required to align \n'
                   'the streamline referential with the true_max_voxel \n'
                   'metric. The file is derived from the out_metrics \n'
                   'parameter with "_tmv_rotation" appended.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_bbox_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not nib.streamlines.is_supported(args.in_centroids):
        parser.error('Invalid input streamline file format (must be trk ' +
                     'or tck): {0}'.format(args.in_centroids))

    out_metrics_no_ext, ext = os.path.splitext(args.out_metrics)

    outputs = [args.out_metrics]
    if args.save_tmv_rotation:
        outputs.append(out_metrics_no_ext + '_tmv_rotation' + ext)

    assert_inputs_exist(parser, [args.in_centroids, args.in_diameters])
    assert_outputs_exist(parser, args, outputs)

    logging.debug('Loading centroid tractogram & diameters')
    in_sft = load_tractogram_with_reference(parser, args, args.in_centroids)
    in_sft.to_voxmm()
    in_sft.to_center()
    # Casting ArraySequence as a list to improve speed
    fibers = list(in_sft.get_streamlines_copy())
    diameters = np.loadtxt(args.in_diameters, dtype=np.float64)
    if args.single_diameter:
        diameters = [diameters[0]]*len(fibers)

    min_ext_dist, min_ext_dist_vect = (
        min_external_distance(fibers, diameters, args.verbose))
    max_voxel_ani, max_voxel_iso = (
        max_voxels(min_ext_dist_vect))
    tmv_rot, tmv_edge = (
        true_max_voxel(min_ext_dist_vect))

    metrics = {
        'min_external_distance': min_ext_dist,
        'max_voxel_anisotropic': max_voxel_ani,
        'max_voxel_isotropic': max_voxel_iso,
        'true_max_voxel': [tmv_edge]*3
    }
    save_dictionary(metrics, args.out_metrics, args.overwrite)

    if args.save_tmv_rotation:
        true_max_voxel_transform = np.r_[np.c_[
            tmv_rot, [0, 0, 0]], [[0, 0, 0, 1]]]
        np.savetxt(out_metrics_no_ext + '_tmv_rotation.txt',
                   true_max_voxel_transform)


if __name__ == '__main__':
    main()
