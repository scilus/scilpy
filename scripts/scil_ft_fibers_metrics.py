#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Computes several metrics from an input centerline tractogram and a text file
containing the diameters of each fiber.

Computed metrics:
    - min_external_distance
        Smallest distance separating two fibers.
    - max_voxel_anisotropic
        Diagonal vector of the largest possible anisotropic voxel that
        would not intersect two fibers.
    - max_voxel_isotropic
        Isotropic version of max_voxel_anisotropic made by using the smallest
        component.
        Ex: max_voxel_anisotropic: (3, 5, 5)
            max_voxel_isotropic: (3, 3, 3)
    - max_voxel_rotated
        Largest possible isotropic voxel if the tractogram is rotated. It is
        obtained by measuring the smallest distance between two fibertubes.
        It is only usable if the entire tractogram is rotated according to
        [rotation_matrix].
        Ex: max_voxel_anisotropic: (1, 0, 0)
            max_voxel_isotropic: (0, 0, 0)
            max_voxel_rotated: (0.5774, 0.5774, 0.5774)
    - rotation_matrix [separate file]
        4D transformation matrix representing the rotation to be applied on
        the tractogram to align max_voxel_rotated with the coordinate system
        (see scil_tractogram_apply_transform.py)."""
import os
import argparse
import logging
import numpy as np
import nibabel as nib

from scilpy.tractanalysis.fibertube_scoring import (
                                        min_external_distance,
                                        max_voxels,
                                        max_voxel_rotated)
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

    p.add_argument('in_centerlines',
                   help='Path to the tractogram file containing the \n'
                   'fibertube centerlines (must be .trk or .tck).')

    p.add_argument('in_diameters',
                   help='Path to a text file containing a list of the \n'
                   'diameters of each fibertube in mm (.txt). Each line \n'
                   'corresponds to the identically numbered centerline.')

    p.add_argument('out_metrics',
                   help='Output file containing the computed metrics \n'
                   '(must be .txt).')

    p.add_argument('--save_rotation_matrix',
                   help='If set, a separate text file will be saved, \n'
                   'containing the transformation matrix required to align \n'
                   'the max_voxel_anisotropic vector with \n'
                   'max_voxel_rotated. The file is derived from the \n'
                   'out_metrics parameter with "_max_voxel_rotation"'
                   'appended.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_bbox_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('numba').setLevel(logging.WARNING)

    if not nib.streamlines.is_supported(args.in_centerlines):
        parser.error('Invalid input streamline file format (must be trk ' +
                     'or tck): {0}'.format(args.in_centerlines))

    out_metrics_no_ext, ext = os.path.splitext(args.out_metrics)

    outputs = [args.out_metrics]
    if args.save_rotation_matrix:
        outputs.append(out_metrics_no_ext + '_max_voxel_rotation' + ext)

    assert_inputs_exist(parser, [args.in_centerlines, args.in_diameters])
    assert_outputs_exist(parser, args, outputs)

    logging.debug('Loading centerline tractogram & diameters')
    in_sft = load_tractogram_with_reference(parser, args, args.in_centerlines)
    in_sft.to_voxmm()
    in_sft.to_center()
    # Casting ArraySequence as a list to improve speed
    fibers = list(in_sft.get_streamlines_copy())
    diameters = np.loadtxt(args.in_diameters, dtype=np.float64)
    if args.single_diameter:
        diameter = diameters if np.ndim(diameters) == 0 else diameters[0]
        diameters = np.full(len(fibers), diameter)

    min_ext_dist, min_ext_dist_vect = (
        min_external_distance(fibers, diameters, args.verbose))
    max_voxel_ani, max_voxel_iso = max_voxels(min_ext_dist_vect)
    mvr_rot, mvr_edge = max_voxel_rotated(min_ext_dist_vect)

    metrics = {
        'min_external_distance': min_ext_dist,
        'max_voxel_anisotropic': max_voxel_ani,
        'max_voxel_isotropic': max_voxel_iso,
        'max_voxel_rotated': [mvr_edge]*3
    }
    save_dictionary(metrics, args.out_metrics, args.overwrite)

    if args.save_rotation_matrix:
        max_voxel_rotated_transform = np.r_[np.c_[
            mvr_rot, [0, 0, 0]], [[0, 0, 0, 1]]]
        np.savetxt(out_metrics_no_ext + '_max_voxel_rotation.txt',
                   max_voxel_rotated_transform)


if __name__ == '__main__':
    main()
