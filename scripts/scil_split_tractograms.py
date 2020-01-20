#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_reference_arg,)

DESCRIPTION = """
    Split a tractogram into multiple files, 2 options available :
    Split into X files, or split into files of Y streamlines
"""


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION)

    p.add_argument('input_tractogram',
                   help='Input filename to split (trk or tck)')
    p.add_argument('output_name',
                   help='Output filename, index will be appended automatically')

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--chunk_size', type=int,
                       help='The maximum number of streamlines per file')

    group.add_argument('--nb_chunk', type=int,
                       help='Divide the file in equal parts')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.input_tractogram)
    out_basename, out_extension = os.path.splitext(args.output_name)

    # Check only the first potential output filename
    assert_outputs_exist(parser, args, [out_basename + '_0' + out_extension])

    sft = load_tractogram_with_reference(parser, args, args.input_tractogram)

    streamlines_count = len(sft.streamlines)

    if args.nb_chunk:
        chunk_size = int(streamlines_count/args.nb_chunk)
        nb_chunk = args.nb_chunk
    else:
        chunk_size = args.chunk_size
        nb_chunk = int(streamlines_count/chunk_size)+1

    # All chunks will be equal except the last one
    chunk_size_array = np.ones((nb_chunk,), dtype=np.int16) * chunk_size
    chunk_size_array[-1] += (streamlines_count - chunk_size * nb_chunk)
    k = 0
    for i in range(nb_chunk):
        streamlines = sft.streamlines[k:(k + chunk_size_array[i])]
        k += chunk_size_array[i]
        new_sft = StatefulTractogram.from_sft(streamlines, sft)

        out_name = '{0}_{1}{2}'.format(out_basename, i, out_extension)
        save_tractogram(new_sft, out_name)


if __name__ == "__main__":
    main()
