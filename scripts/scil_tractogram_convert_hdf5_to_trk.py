#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Save individual connection of an hd5f from
scil_tractogram_segment_bundles_for_connectivity.py.
Useful for quality control and visual inspections.

It can either save all connections, individual connections specified with
edge_keys or connections from specific nodes with node_keys.

With the option save_empty, a label_lists, as a txt file, must be provided.
This option saves existing connections and empty connections.

The output is a directory containing the thousands of connections:
out_dir/
    |-- LABEL1_LABEL1.trk
    |-- LABEL1_LABEL2.trk
    |-- [...]
    |-- LABEL90_LABEL90.trk

Formerly: scil_save_connections_from_hdf5.py
"""

import argparse
import logging
import os

from dipy.io.stateful_tractogram import Space, Origin, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import create_nifti_header
import h5py
import itertools
import numpy as np

from scilpy.io.hdf5 import reconstruct_streamlines_from_hdf5
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_hdf5',
                   help='HDF5 filename (.h5) containing decomposed '
                        'connections.')
    p.add_argument('out_dir',
                   help='Path of the output directory.')

    p.add_argument('--include_dps', action='store_true',
                   help='Include the data_per_streamline the metadata.')

    group = p.add_mutually_exclusive_group()
    group.add_argument('--edge_keys', nargs='+',
                       help='Keys to identify the edges of '
                            'interest (LABEL1_LABEL2).')

    group.add_argument('--node_keys', nargs='+',
                       help='Node keys to identify the '
                            'sub-network of interest.')

    p.add_argument('--save_empty', action='store_true',
                   help='Save empty connections.')
    p.add_argument('--labels_list',
                   help='A txt file containing a list '
                        'saved by the decomposition script.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_hdf5)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)
    if args.save_empty and args.labels_list is None:
        parser.error("The option --save_empty requires --labels_list.")

    with h5py.File(args.in_hdf5, 'r') as hdf5_file:
        if args.save_empty:
            all_labels = np.loadtxt(args.labels_list, dtype='str')
            comb_list = list(itertools.combinations(all_labels, r=2))
            comb_list.extend(zip(all_labels, all_labels))
            keys = [i[0]+'_'+i[1] for i in comb_list]
        else:
            keys = hdf5_file.keys()

        if args.edge_keys is not None:
            selected_keys = [key for key in keys if key in args.edge_keys]
        elif args.node_keys is not None:
            selected_keys = []
            for node in args.node_keys:
                selected_keys.extend([key for key in keys
                                      if key.startswith(node + '_')
                                      or key.endswith('_' + node)])
        else:
            selected_keys = keys

        affine = hdf5_file.attrs['affine']
        dimensions = hdf5_file.attrs['dimensions']
        voxel_sizes = hdf5_file.attrs['voxel_sizes']
        header = create_nifti_header(affine, dimensions, voxel_sizes)
        for key in selected_keys:
            streamlines = reconstruct_streamlines_from_hdf5(hdf5_file[key])

            if len(streamlines) == 0 and not args.save_empty:
                continue

            sft = StatefulTractogram(streamlines, header, Space.VOX,
                                     origin=Origin.TRACKVIS)
            if args.include_dps:
                for dps_key in hdf5_file[key].keys():
                    if dps_key not in ['data', 'offsets', 'lengths']:
                        sft.data_per_streamline[dps_key] = \
                            hdf5_file[key][dps_key]

            save_tractogram(sft, '{}.trk'
                            .format(os.path.join(args.out_dir, key)))


if __name__ == "__main__":
    main()
