#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a tractogram into multiple files, 2 options available :
Split into X files, or split into files of Y streamlines
"""


import argparse
import os

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_reference_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('in_tractogram',
                   help='Tractogram input file name.')
    p.add_argument('out_tractogram',
                   help='Output filename, with extension needed,'
                   'index will be appended automatically.')

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--chunk_size', type=int,
                       help='The maximum number of streamlines per file.')

    group.add_argument('--nb_chunk', type=int,
                       help='Divide the file in equal parts.')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    out_basename, out_extension = os.path.splitext(args.out_tractogram)

    # Check only the first potential output filename
    assert_outputs_exist(parser, args, [out_basename + '_0' + out_extension])

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    streamlines_count = len(sft.streamlines)

    if args.nb_chunk:
        chunk_size = int(streamlines_count/args.nb_chunk)
        nb_chunk = args.nb_chunk
    else:
        chunk_size = args.chunk_size
        nb_chunk = int(streamlines_count/chunk_size)+1

    # All chunks will be equal except the last one
    chunk_sizes = np.ones((nb_chunk,), dtype=np.int16) * chunk_size
    chunk_sizes[-1] += (streamlines_count - chunk_size * nb_chunk)
    curr_count = 0
    for i in range(nb_chunk):
        streamlines = sft.streamlines[curr_count:curr_count + chunk_sizes[i]]
        data_per_streamline = sft.data_per_streamline[curr_count:curr_count
                                                      + chunk_sizes[i]]
        data_per_point = sft.data_per_point[curr_count:curr_count
                                            + chunk_sizes[i]]
        curr_count += chunk_sizes[i]
        new_sft = StatefulTractogram.from_sft(streamlines, sft,
                                              data_per_point=data_per_point,
                                              data_per_streamline=data_per_streamline)

        out_name = '{0}_{1}{2}'.format(out_basename, i, out_extension)
        save_tractogram(new_sft, out_name)


if __name__ == "__main__":
    main()
