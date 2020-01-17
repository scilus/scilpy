#!/usr/bin/env python
# encoding: utf-8


"""
Compute the dice coefficient between two streamline files.

Option to weight the dice coefficient based on the number of tracts passing
through a voxel
"""


import argparse
import json

import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import add_reference_arg, assert_inputs_exist
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("in_bundle1",
                        help="Tractogram filename. Format must be one of \n"
                        "trk, tck, vtk, fib, dpy.")

    parser.add_argument("in_bundle2",
                        help="Tractogram filename. Format must be one of \n"
                        "trk, tck, vtk, fib, dpy.")

    add_reference_arg(parser)

    parser.add_argument("--weighted", action="store_true",
                        help="Weight the dice coefficient based on the \n"
                        "number of tracts passing through a voxel")

    parser.add_argument("--indent", type=int, default=2,
                        help="Indent for json pretty print.")

    return parser


def _load_streamline_count(parser, args, in_bundle):
    tractogram = load_tractogram_with_reference(parser, args, in_bundle)
    tractogram.to_vox()
    tractogram.to_corner()
    streamlines = tractogram.streamlines
    _, dimensions, _, _ = tractogram.space_attribute

    return compute_tract_counts_map(streamlines, dimensions)


def _compute_dice(array1, array2):
    def normalize(array):
        array_float = array.astype(np.float32)
        return (array_float - np.min(array_float)) / np.ptp(array_float)

    if not all([np.any(array1), np.any(array2)]):
        return 0.0

    array1_norm = normalize(array1)
    array2_norm = normalize(array2)

    intersec = (array1_norm * array2_norm).nonzero()
    numerator = np.sum(array1_norm[intersec]) + np.sum(array2_norm[intersec])
    denominator = np.sum(array1_norm) + np.sum(array2_norm)

    dice_coef = float(numerator / denominator)

    if dice_coef > 1.0:
        return 1.0
    else:
        return dice_coef


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(
        parser, args.in_bundle1, optional=args.reference)
    assert_inputs_exist(
        parser, args.in_bundle2, optional=args.reference)

    streamline_count1 = _load_streamline_count(parser, args, args.in_bundle1)
    streamline_count2 = _load_streamline_count(parser, args, args.in_bundle2)

    if not args.weighted:
        streamline_count1 = streamline_count1 > 0
        streamline_count2 = streamline_count2 > 0

    dice_coef = _compute_dice(streamline_count1, streamline_count2)

    print(json.dumps({"dice": dice_coef}, indent=args.indent))


if __name__ == "__main__":
    main()
