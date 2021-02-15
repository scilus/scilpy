#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge a list of tractograms as a single tractogram.
"""

import argparse
import os

from dipy.io.streamline import save_tractogram
from functools import reduce

from scilpy.io.image import assert_same_resolution
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('tractograms', nargs="+",
                        help='List of tractograms.')
    parser.add_argument('out_tractogram',
                        help='Output tractogram.')

    add_overwrite_arg(parser)
    add_reference_arg(parser)
    add_verbose_arg(parser)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Assert inputs and outputs exist
    assert_inputs_exist(parser, args.tractograms)
    assert_outputs_exist(parser, args, args.out_tractogram)

    # Load tractograms for reference check (and their streamlines)
    tractograms = []
    for filename in args.tractograms:
        tractograms += [load_tractogram_with_reference(parser, args, filename)]
        bundle_name, _ = os.path.splitext(os.path.basename(filename))
        print('{}: {} streamlines'.format(
            bundle_name, len(tractograms[-1].streamlines)))

    # Assert tractograms live in the same space
    # Tractograms must be loaded with reference (if needed) beforehand
    assert_same_resolution(tractograms)

    # Sum tractogram list with an accumulator
    full_tractogram = reduce(lambda t1, t2: t1 + t2, tractograms)

    # Print info
    bundle_name, _ = os.path.splitext(os.path.basename(args.out_tractogram))
    print('{}: {} streamlines'.format(
        bundle_name, len(full_tractogram.streamlines)))

    # Save the resulting tractogram
    save_tractogram(full_tractogram, args.out_tractogram)


if __name__ == '__main__':
    main()
