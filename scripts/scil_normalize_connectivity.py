#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import argparse
from copy import copy
import itertools

import nibabel as nib
import numpy as np
from sklearn.neighbors import KDTree

from scilpy.io.utils import (add_overwrite_arg,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,)

    p.add_argument('in_matrix',
                   help='')
    p.add_argument('out_matrix',
                   help='')

    node_p = p.add_argument_group('Node-wise options')
    length = node_p.add_mutually_exclusive_group()
    length.add_argument('--length', metavar='LENGTH_MATRIX',
                        help='CORRECT BIAS FOR WM')
    length.add_argument('--inverse_length', metavar='LENGTH_MATRIX',
                        help='BOOST FOR INT')

    node_p.add_argument('--bundle_volume', metavar='VOLUME_MATRIX',
                        help='WM')

    vol = node_p.add_mutually_exclusive_group()
    vol.add_argument('--parcel_volume', metavar='ATLAS',
                     help='ALL')
    vol.add_argument('--parcel_surface', metavar='ATLAS',
                     help='ALL')

    scaling_p = p.add_argument_group('Scaling options')
    scale = scaling_p.add_mutually_exclusive_group()
    scale.add_argument('--max_at_one', action='store_true',
                       help='')
    scale.add_argument('--sum_to_one', action='store_true',
                       help='')
    scale.add_argument('--log_10', action='store_true',
                       help='')
    scale.add_argument('--log_e', action='store_true',
                       help='')

    add_overwrite_arg(p)

    return p


def approximate_surface_node(atlas, node_id):
    roi = np.zeros(atlas.shape)
    roi[atlas == node_id] = 1
    ind = np.argwhere(roi > 0)
    tree = KDTree(ind)
    count = np.sum(7 - tree.query_radius(ind, r=1.0, count_only=True))

    return count


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_matrix, [args.length,
                                                 args.inverse_length,
                                                 args.bundle_volume,
                                                 args.parcel_volume,
                                                 args.parcel_surface])
    assert_outputs_exist(parser, args, args.out_matrix)

    in_matrix = load_matrix_in_any_format(args.in_matrix)

    # Parcel volume and surface normalization require the atlas
    # This script should be used directly after scil_decompose_connectivity.py
    if args.parcel_volume or args.parcel_surface:
        atlas_filepath = args.parcel_volume if args.parcel_volume \
            else args.parcel_surface
        atlas_img = nib.load(atlas_filepath)
        atlas_data = atlas_img.get_fdata().astype(np.int)

        voxels_size = atlas_img.header.get_zooms()[:3]
        if voxels_size[0] != voxels_size[1] or voxels_size[0] != voxels_size[2]:
            parser.error('Atlas must have an isotropic resolution.')

        voxels_vol = np.prod(atlas_img.header.get_zooms()[:3])
        voxels_sur = np.prod(atlas_img.header.get_zooms()[:2])

        # Excluding background (0)
        labels_list = np.unique(atlas_data)[1:]
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
                factor_list.append(approximate_surface_node(
                    atlas_data, label) * voxels_sur)

        for pos_1, pos_2 in all_comb:
            factor = factor_list[pos_1] + factor_list[pos_2]
            out_matrix[pos_1, pos_2] /= factor
            out_matrix[pos_2, pos_1] /= factor

    # Simple scaling of the whole matrix, facilitate comparison across subject
    if args.max_at_one:
        out_matrix /= np.max(out_matrix)
    elif args.sum_to_one:
        out_matrix /= np.sum(out_matrix)
    elif args.log_10:
        out_matrix = np.log10(out_matrix)
    elif args.log_e:
        out_matrix = np.log(out_matrix)

    save_matrix_in_any_format(args.out_matrix, out_matrix)


if __name__ == "__main__":
    main()
