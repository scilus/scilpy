#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a tractogram into multiple files, 2 options available :
Split into X files, or split into files of Y streamlines.

By default, streamlines to add to each chunk will be chosen randomly.
Optionally, you can split streamlines...
    - sequentially (the first n/nb_chunks streamlines in the first chunk and so
     on).
    - randomly, but per Quickbundles clusters.

Formerly: scil_split_tractogram.py
"""
import argparse
import logging
import os

from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             add_verbose_arg)
from scilpy.tractograms.tractogram_operations import (
    split_sft_sequentially,
    split_sft_randomly,
    split_sft_randomly_per_cluster)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('in_tractogram',
                   help='Tractogram input file name.')
    p.add_argument('out_prefix',
                   help='Prefix for the output tractogram, index will be '
                        'appended \nautomatically (ex, _0.trk), based on '
                        'input type.')

    p.add_argument('--out_dir', default='',
                   help='Put all output tractogram in a specific directory.')

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--chunk_size', type=int,
                       help='The maximum number of streamlines per file.')
    group.add_argument('--nb_chunks', type=int,
                       help='Divide the file in equal parts.')

    group2 = p.add_mutually_exclusive_group()
    group2.add_argument(
        '--split_per_cluster', action='store_true',
        help='If set, splitting will be done per cluster (computed with \n'
             'Quickbundles) to ensure that at least some streamlines are \n'
             'kept from each bundle in each chunk. Else, random splitting is\n'
             'performed (default).')
    group2.add_argument(
        '--do_not_randomize', action='store_true',
        help="If set, splitting is done sequentially through the original \n"
             "sft instead of using random indices.")
    p.add_argument('--qbx_thresholds', nargs='+', type=float,
                   default=[40, 30, 20], metavar='t',
                   help="If you chose option '--split_per_cluster', you may "
                        "set the \nQBx threshold value(s) here. Default: "
                        "%(default)s")

    p.add_argument('--seed', default=None, type=int,
                   help='Use a specific random seed for the subsampling.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    _, out_extension = os.path.splitext(args.in_tractogram)

    assert_output_dirs_exist_and_empty(parser, args, [], optional=args.out_dir)
    # Check only the first potential output filename, we don't know how many
    # there are yet.
    assert_outputs_exist(parser, args, os.path.join(
        args.out_dir, '{}_0{}'.format(args.out_prefix, out_extension)))

    logging.info("Loading sft.")
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    streamlines_count = len(sft.streamlines)

    if args.nb_chunks:
        chunk_size = int(streamlines_count/args.nb_chunks)
        nb_chunks = args.nb_chunks
    else:
        chunk_size = args.chunk_size
        nb_chunks = int(streamlines_count/chunk_size)+1

    # Check other outputs
    out_names = ['{0}_{1}{2}'.format(args.out_prefix, i, out_extension) for
                 i in range(nb_chunks)]
    assert_outputs_exist(parser, args,
                         [os.path.join(args.out_dir, out_names[i]) for i in
                          range(1, nb_chunks)])

    # All chunks will be equal except the last one
    chunk_sizes = np.ones((nb_chunks,), dtype=np.int16) * chunk_size
    chunk_sizes[-1] += (streamlines_count - chunk_size * nb_chunks)

    if args.do_not_randomize:
        sfts = split_sft_sequentially(sft, chunk_sizes)
    elif args.split_per_cluster:
        # With this version, will contain an additional sft with non-included
        # streamlines. Should be of size close to 0. Not using it.
        sfts = split_sft_randomly_per_cluster(
            sft, chunk_sizes, args.seed, args.qbx_thresholds)
    else:
        sfts = split_sft_randomly(sft, chunk_sizes, args.seed)

    for i in range(nb_chunks):
        out_name = os.path.join(args.out_dir, out_names[i])
        save_tractogram(sfts[i], out_name)


if __name__ == "__main__":
    main()
