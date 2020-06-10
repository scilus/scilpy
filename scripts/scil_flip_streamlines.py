#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flip streamlines locally around specific axes.

IMPORTANT: this script should only be used in case of absolute necessity.
It's better to fix the real tools than to force flipping streamlines to
have them fit in the tools.
"""

import argparse

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_reference_arg,
                             add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file.')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file.')

    p.add_argument('axes',
                   choices=['x', 'y', 'z'], nargs='+',
                   help='The axes you want to flip. eg: to flip the x '
                        'and y axes use: x y.')

    add_reference_arg(p)
    add_overwrite_arg(p)
    return p


def get_axis_flip_vector(flip_axes):
    flip_vector = np.ones(3)
    if 'x' in flip_axes:
        flip_vector[0] = -1.0
    if 'y' in flip_axes:
        flip_vector[1] = -1.0
    if 'z' in flip_axes:
        flip_vector[2] = -1.0

    return flip_vector


def get_shift_vector(sft):
    dims = sft.space_attributes[1]
    shift_vector = -1.0 * (np.array(dims) / 2.0)

    return shift_vector


def flip_sft(sft, flip_axes):
    flip_vector = get_axis_flip_vector(flip_axes)
    shift_vector = get_shift_vector(sft)

    flipped_streamlines = []

    streamlines = sft.streamlines

    for streamline in streamlines:
        mod_streamline = streamline + shift_vector
        mod_streamline *= flip_vector
        mod_streamline -= shift_vector
        flipped_streamlines.append(mod_streamline)

    new_sft = StatefulTractogram.from_sft(flipped_streamlines, sft,
                                          data_per_point=sft.data_per_point,
                                          data_per_streamline=sft.data_per_streamline)
    return new_sft


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_vox()
    sft.to_corner()

    new_sft = flip_sft(sft, args.axes)
    save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
