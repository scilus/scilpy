#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_reference)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Assign an hexadecimal RGB color to a Trackvis TRK '
                    'tractogram. The hexadecimal RGB color should be '
                    'formatted as 0xRRGGBB',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Tractogram.')

    add_reference(p)

    p.add_argument('out_tractogram',
                   help='Colored TRK tractogram.')
    p.add_argument('color',
                   help='Hexadecimal RGB color (ie. 0xRRGGBB).')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    if not args.out_tractogram.endswith('.trk'):
        parser.error('Output file needs to end with .trk.')

    if len(args.color) != 8:
        parser.error('Hexadecimal RGB color should be formatted as 0xRRGGBB.')

    color_int = int(args.color, 0)
    red = color_int >> 16
    green = (color_int & 0x00FF00) >> 8
    blue = color_int & 0x0000FF

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    sft.data_per_point["color"] = [np.tile([red, green, blue],
                                   (len(i), 1)) for i in sft.streamlines]

    sft = StatefulTractogram(sft.streamlines, sft, Space.RASMM,
                             data_per_point=sft.data_per_point)

    save_tractogram(sft, args.out_tractogram)


if __name__ == '__main__':
    main()
