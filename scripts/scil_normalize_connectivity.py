#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize a connectivity matrix coming from scil_decompose_connectivity.py.
3 categories of normalization are available:
-- Edge attributes
 - length: Multiply each edge by the average bundle length.
   Compensate for far away connections when using interface seeding.
   Cannot be used with inverse_length.

 - inverse_length: Divide each edge by the average bundle length.
   Compensate for big connections when using white matter seeding.
   Cannot be used with length.

 - bundle_volume: Divide each edge by the average bundle length.
   Compensate for big connections when using white matter seeding.

-- Node attributes (Mutually exclusive)
 - parcel_volume: Divide each edge by the sum of node volume.
   Compensate for the likelihood of ending in the node.
   Compensate seeding bias when using interface seeding.

 - parcel_surface: Divide each edge by the sum of the node surface.
   Compensate for the likelihood of ending in the node.
   Compensate for seeding bias when using interface seeding.

-- Matrix scaling (Mutually exclusive)
 - max_at_one: Maximum value of the matrix will be set to one.
 - sum_to_one: Ensure the sum of all edges weight is one
 - log_10: Apply a base 10 logarithm to all edges weight

The volume and length matrix should come from the
scil_decompose_connectivity.py script.

A review of the type of normalization is available in:
Colon-Perez, Luis M., et al. "Dimensionless, scale-invariant, edge weight
metric for the study of complex structural networks." PLOS one 10.7 (2015).

However, the proposed weighting of edge presented in this publication is not
implemented.
"""

import argparse
from copy import copy
import itertools

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.image.operations import normalize_max, normalize_sum, base_10_log
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format)
from scilpy.tractanalysis.reproducibility_measures import \
    approximate_surface_node


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,)

    p.add_argument('in_matrix',
                   help='Input connectivity matrix. This is typically a '
                        'streamline_count matrix (.npy).')
    p.add_argument('out_matrix',
                   help='Output normalized matrix (.npy).')

    edge_p = p.add_argument_group('Edge-wise options')
    length = edge_p.add_mutually_exclusive_group()
    length.add_argument('--length', metavar='LENGTH_MATRIX',
                        help='Length matrix used for '
                        'edge-wise multiplication.')
    length.add_argument('--inverse_length', metavar='LENGTH_MATRIX',
                        help='Length matrix used for edge-wise division.')
    edge_p.add_argument('--bundle_volume', metavar='VOLUME_MATRIX',
                        help='Volume matrix used for edge-wise division.')

    vol = edge_p.add_mutually_exclusive_group()
    vol.add_argument('--parcel_volume', nargs=2,
                     metavar=('ATLAS', 'LABELS_LIST'),
                     help='Atlas and labels list for edge-wise division.')
    vol.add_argument('--parcel_surface', nargs=2,
                     metavar=('ATLAS', 'LABELS_LIST'),
                     help='Atlas and labels list for edge-wise division.')

    scaling_p = p.add_argument_group('Scaling options')
    scale = scaling_p.add_mutually_exclusive_group()
    scale.add_argument('--max_at_one', action='store_true',
                       help='Scale matrix with maximum value at one.')
    scale.add_argument('--sum_to_one', action='store_true',
                       help='Scale matrix with sum of all elements at one.')
    scale.add_argument('--log_10', action='store_true',
                       help='Apply a base 10 logarithm to the matrix.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_matrix, [args.length,
                                                 args.inverse_length,
                                                 args.bundle_volume])
    assert_outputs_exist(parser, args, args.out_matrix)

    in_matrix = load_matrix_in_any_format(args.in_matrix)

    # Parcel volume and surface normalization require the atlas
    # This script should be used directly after scil_decompose_connectivity.py
    if args.parcel_volume or args.parcel_surface:
        atlas_tuple = args.parcel_volume if args.parcel_volume \
            else args.parcel_surface
        atlas_filepath, labels_filepath = atlas_tuple
        assert_inputs_exist(parser, [atlas_filepath, labels_filepath])

        atlas_img = nib.load(atlas_filepath)
        atlas_data = get_data_as_labels(atlas_img)

        voxels_size = atlas_img.header.get_zooms()[:3]
        if voxels_size[0] != voxels_size[1] \
           or voxels_size[0] != voxels_size[2]:
            parser.error('Atlas must have an isotropic resolution.')

        voxels_vol = np.prod(atlas_img.header.get_zooms()[:3])
        voxels_sur = np.prod(atlas_img.header.get_zooms()[:2])

        # Excluding background (0)
        labels_list = np.loadtxt(labels_filepath)
        if len(labels_list) != in_matrix.shape[0] \
                and len(labels_list) != in_matrix.shape[1]:
            parser.error('Atlas should have the same number of label as the '
                         'input matrix.')

    # Normalization can be combined together
    out_matrix = in_matrix
    if args.length:
        length_mat = load_matrix_in_any_format(args.length)
        out_matrix[length_mat > 0] *= length_mat[length_mat > 0]
    elif args.inverse_length:
        length_mat = load_matrix_in_any_format(args.inverse_length)
        out_matrix[length_mat > 0] /= length_mat[length_mat > 0]

    if args.bundle_volume:
        volume_mat = load_matrix_in_any_format(args.bundle_volume)
        out_matrix[volume_mat > 0] /= volume_mat[volume_mat > 0]

    # Node-wise computation are necessary for this type of normalize
    if args.parcel_volume or args.parcel_surface:
        out_matrix = copy(in_matrix)
        pos_list = range(len(labels_list))
        all_comb = list(itertools.combinations(pos_list, r=2))
        all_comb.extend(zip(pos_list, pos_list))

        # Prevent useless computions for approximate_surface_node()
        factor_list = []
        for label in labels_list:
            if args.parcel_volume:
                factor_list.append(np.count_nonzero(
                    atlas_data == label) * voxels_vol)
            else:
                if np.count_nonzero(atlas_data == label):
                    roi = np.zeros(atlas_data.shape)
                    roi[atlas_data == label] = 1
                    factor_list.append(
                        approximate_surface_node(roi) * voxels_sur)
                else:
                    factor_list.append(0)

        for pos_1, pos_2 in all_comb:
            factor = factor_list[pos_1] + factor_list[pos_2]
            if abs(factor) > 0.001:
                out_matrix[pos_1, pos_2] /= factor
                out_matrix[pos_2, pos_1] /= factor

    # Load as image
    ref_matrix = nib.Nifti1Image(in_matrix, np.eye(4))
    # Simple scaling of the whole matrix, facilitate comparison across subject
    if args.max_at_one:
        out_matrix = nib.Nifti1Image(out_matrix, np.eye(4))
        out_matrix = normalize_max([out_matrix], ref_matrix)
    elif args.sum_to_one:
        out_matrix = nib.Nifti1Image(out_matrix, np.eye(4))
        out_matrix = normalize_sum([out_matrix], ref_matrix)
    elif args.log_10:
        out_matrix = nib.Nifti1Image(out_matrix, np.eye(4))
        out_matrix = base_10_log([out_matrix], ref_matrix)

    save_matrix_in_any_format(args.out_matrix, out_matrix)


if __name__ == "__main__":
    main()
