#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Save connections as TRK to HDF5.

This script is useful to convert a set of connections or bundles to a single
HDF5 file. The HDF5 file will contain a group for each input file, with the
streamlines stored in the specified space and origin (keep the default if you
are going to use the connectivity scripts in scilpy).

To make a file compatible with scil_tractogram_commit.py or
scil_connectivity_compute_matrices.py you will have to follow this nomenclature
for the input files:
in_dir/
    |-- LABEL1_LABEL1.trk
    |-- LABEL1_LABEL2.trk
    |-- [...]
    |-- LABEL90_LABEL90.trk
The value of first labels should be smaller or equal to the second labels.
Connectivity scripts in scilpy only consider the upper triangular part of the
connectivity matrix.

By default, ignores the empty connections. To save them, use --save_empty.
Note that data_per_point is never included.
"""

import argparse
import logging
import os

from dipy.io.stateful_tractogram import Space, Origin
from dipy.io.utils import is_header_compatible
import h5py

from scilpy.io.hdf5 import (construct_hdf5_header,
                            construct_hdf5_group_from_streamlines)
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_bundles', nargs='+',
                   help='Path of the input connection(s) or bundle(s).')
    p.add_argument('out_hdf5',
                   help='Output HDF5 filename (.h5).')

    p.add_argument('--stored_space', choices=['rasmm', 'voxmm', 'vox'],
                   default='vox',
                   help='Space convention in which the streamlines are stored '
                        '[%(default)s].')
    p.add_argument('--stored_origin', choices=['nifti', 'trackvis'],
                   default='trackvis',
                   help='Voxel origin convention in which the streamlines are '
                        'stored [%(default)s].')

    p.add_argument('--include_dps', action='store_true',
                   help='Include the data_per_streamline the metadata.')
    p.add_argument('--save_empty', action='store_true',
                   help='Save empty connections.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bundles)
    assert_outputs_exist(parser, args, args.out_hdf5)

    ref_sft = load_tractogram_with_reference(parser, args, args.in_bundles[0])

    # Convert STR to the Space and Origin ENUMS
    target_space = Space[args.stored_space.upper()]
    target_origin = Origin[args.stored_origin.upper()]
    with h5py.File(args.out_hdf5, 'w') as hdf5_file:
        for i, in_bundle in enumerate(args.in_bundles):
            in_basename = os.path.splitext(os.path.basename(in_bundle))[0]
            curr_sft = load_tractogram_with_reference(parser, args, in_bundle)
            if len(curr_sft) == 0 and not args.save_empty:
                logging.warning(f"Skipping {in_bundle} because it is empty")
                continue

            if not is_header_compatible(ref_sft, curr_sft):
                parser.error(f"Header of {in_bundle} is not compatible")

            curr_sft.to_space(target_space)
            curr_sft.to_origin(target_origin)

            if i == 0:
                construct_hdf5_header(hdf5_file, ref_sft)
            group = hdf5_file.create_group(in_basename)
            dps = curr_sft.data_per_streamline if args.include_dps else {}
            construct_hdf5_group_from_streamlines(group, curr_sft.streamlines,
                                                  dps=dps)


if __name__ == "__main__":
    main()
