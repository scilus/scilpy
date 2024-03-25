#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flip streamlines locally around specific axes.

IMPORTANT: this script should only be used in case of absolute necessity.
It's better to fix the real tools than to force flipping streamlines to
have them fit in the tools.

Formerly: scil_flip_streamlines.py
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_reference_arg,
                             add_verbose_arg,
                             add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tractograms.tractogram_operations import flip_sft


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
    add_verbose_arg(p)
    add_overwrite_arg(p)
    
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_vox()
    sft.to_corner()

    new_sft = flip_sft(sft, args.axes)
    save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
