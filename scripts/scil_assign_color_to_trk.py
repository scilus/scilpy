#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import nibabel as nib
import numpy as np

from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Assign an hexadecimal RGB color to a Trackvis TRK '
                    'tractogram. The hexadecimal RGB color should be '
                    'formatted as 0xRRGGBB',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('input',
                   help='Tractogram.')
    p.add_argument('output',
                   help='Colored TRK tractogram.')
    p.add_argument('color',
                   help='Hexadecimal RGB color (ie. 0xRRGGBB).')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.input)
    assert_outputs_exist(parser, args, args.output)

    if not args.output.endswith('.trk'):
        parser.error('Output file needs to end with .trk.')

    if len(args.color) != 8:
        parser.error('Hexadecimal RGB color should be formatted as 0xRRGGBB.')

    color_int = int(args.color, 0)
    red = color_int >> 16
    green = (color_int & 0x00FF00) >> 8
    blue = color_int & 0x0000FF

    tractogram_file = nib.streamlines.load(args.input)
    tractogram_file.tractogram.data_per_point["color"] = [
            np.tile([red, green, blue],
                    (len(i), 1)) for i in tractogram_file.streamlines
        ]

    nib.streamlines.save(tractogram_file.tractogram, args.output,
                         header=tractogram_file.header)


if __name__ == '__main__':
    main()
