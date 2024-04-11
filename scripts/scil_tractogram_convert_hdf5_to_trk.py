#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Save connections of a hdf5 created with
>> scil_tractogram_segment_bundles_for_connectivity.py.

Useful for quality control and visual inspections.

It can either save all connections (default), individual connections specified
with --edge_keys or connections from specific nodes specified with --node_keys.

With the option --save_empty, a label_lists, as a txt file, must be provided.
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

from dipy.io.streamline import save_tractogram
import h5py
import itertools
import numpy as np

from scilpy.io.hdf5 import reconstruct_sft_from_hdf5
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
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
    group.add_argument('--edge_keys', nargs='+', metavar='LABEL1_LABEL2',
                       help='Keys to identify the edges (connections) of '
                            'interest.')
    group.add_argument('--node_keys', nargs='+', metavar='NODE',
                       help='Node keys to identify the sub-networks of '
                            'interest.\nEquivalent to adding any --edge_keys '
                            'node_LABEL2 or LABEL2_node.')

    p.add_argument('--save_empty', metavar='labels_list', dest='labels_list',
                   help='Save empty connections. Then, the list of possible '
                        'connections is \nnot found from the hdf5 but '
                        'inferred from labels_list, a txt file \ncontaining '
                        'a list of nodes saved by the decomposition script.\n'
                        '*If used together with edge_keys or node_keys, the '
                        'provided nodes must \nexist in labels_list.')

    add_verbose_arg(p)
    add_overwrite_arg(p, will_delete_dirs=True)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, args.in_hdf5, args.labels_list)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)

    # Processing
    with h5py.File(args.in_hdf5, 'r') as hdf5_file:
        all_hdf5_keys = list(hdf5_file.keys())

        if args.labels_list:
            all_labels = np.loadtxt(args.labels_list, dtype='str')
            comb_list = list(itertools.combinations(all_labels, r=2))
            comb_list.extend(zip(all_labels, all_labels))
            all_keys = [i[0]+'_'+i[1] for i in comb_list]
            keys_origin = "the labels_list file's labels combination"
            allow_empty = True
        else:
            all_keys = all_hdf5_keys
            keys_origin = "the hdf5 stored keys"
            allow_empty = False

        if args.edge_keys is not None:
            selected_keys = args.edge_keys

            # Check that all selected_keys exist.
            impossible_keys = np.setdiff1d(selected_keys, all_keys)
            if len(impossible_keys) > 0:
                parser.error("The following key(s) to not exist in {}: {}\n"
                             "Please verify your --edge_keys."
                             .format(keys_origin, impossible_keys))

        elif args.node_keys is not None:
            selected_keys = []
            for node in args.node_keys:
                selected_keys.extend([key for key in all_keys
                                      if key.startswith(node + '_')
                                      or key.endswith('_' + node)])
            logging.debug("All keys found for provided nodes are: {}"
                          .format(selected_keys))
        else:
            selected_keys = all_keys
            logging.debug("All keys are: {}".format(selected_keys))

        for key in selected_keys:
            sft, _ = reconstruct_sft_from_hdf5(hdf5_file, key,
                                               load_dps=args.include_dps,
                                               allow_empty=allow_empty)
            save_tractogram(sft, '{}.trk'
                            .format(os.path.join(args.out_dir, key)))


if __name__ == "__main__":
    main()
