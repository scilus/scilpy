#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assign an hexadecimal RGB color to a Trackvis TRK tractogram.
The hexadecimal RGB color should be formatted as 0xRRGGBB or
"#RRGGBB".

Saves the RGB values in the data_per_point (color_x, color_y, color_z).

If called with .tck, the output will always be .trk, because data_per_point has
no equivalent in tck file.

Formerly: scil_assign_uniform_color_to_tractograms.py
"""

import argparse
import json
import logging
import os

from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg,
                             add_reference_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractograms', nargs='+',
                   help='Input tractograms (.trk or .tck).')

    g1 = p.add_argument_group(title='Coloring Methods')
    p1 = g1.add_mutually_exclusive_group()
    p1.add_argument('--fill_color',
                    help='Can be either hexadecimal (ie. "#RRGGBB" '
                         'or 0xRRGGBB).')
    p1.add_argument('--dict_colors',
                    help='Dictionnary mapping basename to color.\n'
                         'Same convention as --fill_color.')

    g2 = p.add_argument_group(title='Output options')
    p2 = g2.add_mutually_exclusive_group()
    p2.add_argument('--out_suffix', default='colored',
                    help='Specify suffix to append to input basename.')
    p2.add_argument('--out_tractogram',
                    help='Output filename of colored tractogram (.trk).\n'
                         'Cannot be used with --dict_colors.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)

    if len(args.in_tractograms) > 1 and args.out_tractogram:
        parser.error('Using multiple inputs, use --out_suffix.')

    assert_inputs_exist(parser, args.in_tractograms)

    if args.out_suffix:
        if args.out_tractogram:
            args.out_suffix = ''
        elif args.out_suffix[0] != '_':
            args.out_suffix = '_'+args.out_suffix

    out_filenames = []
    for filename in args.in_tractograms:
        base, ext = os.path.splitext(filename) if args.out_tractogram is None \
            else os.path.splitext(args.out_tractogram)
        if not ext == '.trk':
            logging.warning('Input is TCK file, will be converted to TRK.')
        out_filenames.append('{}{}{}'.format(base, args.out_suffix, '.trk'))
    assert_outputs_exist(parser, args, out_filenames)

    for i, filename in enumerate(args.in_tractograms):
        sft = load_tractogram_with_reference(parser, args, filename)
        base, ext = os.path.splitext(filename)
        out_filename = out_filenames[i]
        pos = base.index('__') if '__' in base else -2
        base = base[pos+2:]
        color = None
        if args.dict_colors:
            with open(args.dict_colors, 'r') as data:
                dict_colors = json.load(data)
            # Supports variation from rbx-flow
            for key in dict_colors.keys():
                if key in base:
                    color = dict_colors[key]
        elif args.fill_color is not None:
            color = args.fill_color

        if color is None:
            color = '0x000000'

        if len(color) == 7:
            args.fill_color = '0x' + args.fill_color.lstrip('#')

        if len(color) == 8:
            color_int = int(color, 0)
            red = color_int >> 16
            green = (color_int & 0x00FF00) >> 8
            blue = color_int & 0x0000FF
        else:
            parser.error('Hexadecimal RGB color should be formatted as '
                         '"#RRGGBB" or 0xRRGGBB.')
        tmp = [np.tile([red, green, blue], (len(i), 1))
               for i in sft.streamlines]
        sft.data_per_point['color'] = tmp
        save_tractogram(sft, out_filename)


if __name__ == '__main__':
    main()
