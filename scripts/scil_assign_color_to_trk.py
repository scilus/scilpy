#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assign an hexadecimal RGB color to a Trackvis TRK tractogram.
The hexadecimal RGB color should be formatted as 0xRRGGBB or
"#RRGGBB".

Saves the RGB values in the data_per_point (color_x, color_y, color_z).
"""

import argparse
import os

from dipy.io.streamline import save_tractogram, load_tractogram
import json
import numpy as np

from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractograms', nargs='+',
                   help='Tractograms.')
    p1 = p.add_mutually_exclusive_group()
    p1.add_argument('--fill_color',
                    help='Can be either hexadecimal (ie. "#RRGGBB" '
                         'or 0xRRGGBB).')
    p1.add_argument('--dict_colors',
                    help='Dictionnary mapping basename to color.'
                         'Same convention as --color.')
    p2 = p.add_mutually_exclusive_group()
    p2.add_argument('--out_suffix', default='colored',
                    help='Specify suffix to append to input')
    p2.add_argument('--out_name',
                    help='Colored TRK tractogram.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractograms)
    if args.out_suffix:
        if args.out_name:
            args.out_suffix = ''
        else:
            args.out_suffix = '_'+args.out_suffix

    if len(args.in_tractograms) > 1 and args.out_name:
        parser.error('Using multiple inputs, use --out_suffix.')
    out_filenames = []
    for filename in args.in_tractograms:
        base, ext = os.path.splitext(filename) if args.out_name is None \
            else os.path.splitext(args.out_name)
        if not ext == '.trk':
            parser.error('Output file needs to end with .trk.')
        out_filenames.append('{}{}{}'.format(base, args.out_suffix, ext))
    assert_outputs_exist(parser, args, out_filenames)

    for i, filename in enumerate(args.in_tractograms):
        base, ext = os.path.splitext(filename)
        out_filename = out_filenames[i]
        pos = base.index('__') if '__' in base else -2
        base = base[pos+2:]
        if args.dict_colors:
            with open(args.dict_colors, 'r') as data:
                dict_colors = json.load(data)
            color = dict_colors[base]
        else:
            color = args.fill_color

        if len(color) == 7:
            args.fill_color = '0x' + args.fill_color.lstrip('#')

        if len(color) == 8:
            color_int = int(color, 0)
            red = color_int >> 16
            green = (color_int & 0x00FF00) >> 8
            blue = color_int & 0x0000FF
        else:
            parser.error('Hexadecimal RGB color should be formatted as "#RRGGBB"'
                         ' or 0xRRGGBB.')

        sft = load_tractogram(filename, 'same')
        sft.data_per_point["color"] = [np.tile([red, green, blue],
                                               (len(i), 1)) for i in sft.streamlines]
        save_tractogram(sft, out_filename)


if __name__ == '__main__':
    main()
