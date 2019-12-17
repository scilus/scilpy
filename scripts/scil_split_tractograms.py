#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)

DESCRIPTION = """
    Split a tractogram into multiple files, 2 options available :
    Split into X files, or split into files of Y streamlines
"""


def buildArgsParser():
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

    add_overwrite_arg(p)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.input_tractogram])
    _, in_extension = os.path.splitext(args.input_tractogram)
    out_basename, out_extension = os.path.splitext(args.output_name)

    # Check only the first potential output filename
    assert_outputs_exist(parser, args, [out_basename + '_0' + out_extension])
    if not in_extension == out_extension:
        parser.error('The input and output files must have the same extension')

    input_tractogram = nib.streamlines.load(args.input_tractogram,
                                            lazy_load=True)
    if in_extension == '.trk':
        streamlines_count = input_tractogram.header['nb_streamlines']
    elif in_extension == '.tck':
        streamlines_count = int(input_tractogram.header['count'])

    if args.nb_chunk:
        chunk_size = int(streamlines_count/args.nb_chunk)
        nb_chunk = args.nb_chunk
    else:
        chunk_size = args.chunk_size
        nb_chunk = int(streamlines_count/chunk_size)+1

    # All chunks will be equal except the last one
    chunk_size_array = np.ones((nb_chunk,), dtype=np.int16) * chunk_size
    chunk_size_array[-1] += (streamlines_count - chunk_size*nb_chunk)
    iterator = iter(input_tractogram.streamlines)
    for i in range(nb_chunk):
        streamlines = read_next(iterator, chunk_size_array[i])
        new_tractogram = nib.streamlines.Tractogram(streamlines,
                                                    affine_to_rasmm=np.eye(4))

        out_name = '{0}_{1}{2}'.format(out_basename, i, out_extension)
        nib.streamlines.save(new_tractogram, out_name,
                             header=input_tractogram.header)


def read_next(iterator, n):
    """Reads and returns 'n' next streamlines (or less if not enough left
    to read) from the current iterator's position."""
    streamlines = []
    for _ in range(n):
        try:
            streamlines.append(next(iterator))
        except StopIteration:
            break
    return streamlines


if __name__ == "__main__":
    main()
