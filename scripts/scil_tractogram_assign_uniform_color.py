#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assign an hexadecimal RGB color to one or more Trackvis (.trk) tractogram.
(If called with .tck, the output will always be .trk, because data_per_point
has no equivalent in tck file.)

Saves the RGB values in the data_per_point 'color' with values
(color_x, color_y, color_z).

The hexadecimal RGB color should be formatted as 0xRRGGBB or "#RRGGBB".

See also: scil_tractogram_assign_custom_color.py

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
                             add_reference_arg, assert_headers_compatible)
from scilpy.viz.color import format_hexadecimal_color_to_rgb


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractograms', nargs='+',
                   help='Input tractograms (.trk or .tck).')

    g1 = p.add_argument_group(title='Coloring Methods')
    p1 = g1.add_mutually_exclusive_group(required=True)
    p1.add_argument('--fill_color', metavar='str',
                    help='Can be hexadecimal (ie. either "#RRGGBB" '
                         'or 0xRRGGBB).')
    p1.add_argument('--dict_colors', metavar='file.json',
                    help="Json file: dictionnary mapping each tractogram's "
                         "basename to a color.\nDo not put your file's "
                         "extension in your dict.\n"
                         "Same convention as --fill_color.")

    g2 = p.add_argument_group(title='Output options')
    p2 = g2.add_mutually_exclusive_group(required=True)
    p2.add_argument('--out_suffix', nargs='?', const='colored',
                    metavar='suffix',
                    help='Specify suffix to append to input basename.\n'
                         'Mandatory choice if you run this script on multiple '
                         'tractograms.\nMandatory choice with --dict_colors.\n'
                         '[%(default)s]')
    p2.add_argument('--out_tractogram', metavar='file.trk',
                    help='Output filename of colored tractogram (.trk).')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    if len(args.in_tractograms) > 1 and args.out_tractogram:
        parser.error('Using multiple inputs, use --out_suffix.')
    if args.dict_colors and args.out_tractogram:
        parser.error('Using --dict_colors, use --out_suffix.')

    assert_inputs_exist(parser, args.in_tractograms, args.reference)
    assert_headers_compatible(parser, args.in_tractograms,
                              reference=args.reference)

    if args.out_suffix and args.out_suffix[0] != '_':
        args.out_suffix = '_' + args.out_suffix

    if args.out_tractogram:
        out_filenames = [args.out_tractogram]
        _, ext = os.path.splitext(args.out_tractogram)
        if not ext == '.trk':
            parser.error("--out_tractogram must be a .trk file.")
    else:  # args.out_suffix
        out_filenames = []
        for filename in args.in_tractograms:
            base, ext = os.path.splitext(filename)
            if not ext == '.trk':
                logging.warning('Input is a .tck file, but output will be a '
                                '.trk file.')
            out_filenames.append('{}{}{}'
                                 .format(base, args.out_suffix, '.trk'))
    assert_outputs_exist(parser, args, out_filenames)

    # Loading (except tractograms, in loop)
    dict_colors = None
    if args.dict_colors:
        with open(args.dict_colors, 'r') as data:
            dict_colors = json.load(data)

    # Processing
    for i, filename in enumerate(args.in_tractograms):
        color = None

        sft = load_tractogram_with_reference(parser, args, filename)

        if args.dict_colors:
            base, ext = os.path.splitext(filename)
            pos = base.index('__') if '__' in base else -2
            base = base[pos + 2:]

            for key in dict_colors.keys():
                if key in base:
                    color = dict_colors[key]
            if color is None:
                parser.error("Basename of file {} ({}) not found in your "
                             "dict_colors keys.".format(filename, base))
        else:  # args.fill_color is not None:
            color = args.fill_color

        red, green, blue = format_hexadecimal_color_to_rgb(color)

        tmp = [np.tile([red, green, blue], (len(i), 1))
               for i in sft.streamlines]
        sft.data_per_point['color'] = tmp

        # Saving
        save_tractogram(sft, out_filenames[i])


if __name__ == '__main__':
    main()
