#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize a connectivity matrix coming from
scil_tractogram_segment_bundles_for_connectivity.py.
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
scil_tractogram_segment_bundles_for_connectivity.py script.

A review of the type of normalization is available in:
Colon-Perez, Luis M., et al. "Dimensionless, scale-invariant, edge weight
metric for the study of complex structural networks." PLOS one 10.7 (2015).

However, the proposed weighting of edge presented in this publication is not
implemented.

Formerly: scil_normalize_connectivity.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.connectivity.connectivity_tools import \
    normalize_matrix_from_values, normalize_matrix_from_parcel
from scilpy.image.volume_math import normalize_max, normalize_sum, base_10_log
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format)


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
                        help='Length matrix used for edge-wise '
                             'multiplication.')
    length.add_argument('--inverse_length', metavar='LENGTH_MATRIX',
                        help='Length matrix used for edge-wise division.')
    edge_p.add_argument('--bundle_volume', metavar='VOLUME_MATRIX',
                        help='Volume matrix used for edge-wise division.')

    vol = edge_p.add_mutually_exclusive_group()
    # toDo. Same description for the two options
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

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_matrix, [args.length,
                                                 args.inverse_length,
                                                 args.bundle_volume])
    assert_outputs_exist(parser, args, args.out_matrix)

    atlas_filepath = None
    labels_filepath = None
    if args.parcel_volume or args.parcel_surface:
        atlas_tuple = args.parcel_volume if args.parcel_volume \
            else args.parcel_surface
        atlas_filepath, labels_filepath = atlas_tuple
        assert_inputs_exist(parser, [atlas_filepath, labels_filepath])

    in_matrix = load_matrix_in_any_format(args.in_matrix)

    # Normalization can be combined.
    out_matrix = in_matrix
    if args.length or args.inverse_length:
        inverse = args.inverse_length is not None
        matrix_file = args.inverse_length if inverse else args.length
        length_matrix = load_matrix_in_any_format(matrix_file)
        out_matrix = normalize_matrix_from_values(
            out_matrix, length_matrix, inverse)

    if args.bundle_volume:
        volume_mat = load_matrix_in_any_format(args.bundle_volume)
        out_matrix = normalize_matrix_from_values(
            out_matrix, volume_mat, inverse=True)

    # Node-wise computation are necessary for this type of normalize
    # Parcel volume and surface normalization require the atlas
    # This script should be used directly after
    # scil_tractogram_segment_bundles_for_connectivity.py
    if args.parcel_volume or args.parcel_surface:
        atlas_img = nib.load(atlas_filepath)
        labels_list = np.loadtxt(labels_filepath)
        out_matrix = normalize_matrix_from_parcel(
            out_matrix, atlas_img, labels_list,
            parcel_from_volume=args.parcel_volume)

    # Save as image
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
