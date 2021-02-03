#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assign an hexadecimal RGB color to a Trackvis TRK tractogram.
The hexadecimal RGB color should be formatted as 0xRRGGBB or
"#RRGGBB".

Saves the RGB values in the data_per_point (color_x, color_y, color_z).

If called with .tck, the output will always be .trk, because data_per_point has
no equivalent in tck file.
"""

import argparse
import logging
import os

from dipy.io.streamline import save_tractogram
import json
import numpy as np
import matplotlib.pyplot as plt

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_reference_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractograms', nargs='+',
                   help='Input tractograms (.trk or .tck).')

    g1 = p.add_argument_group(title='Coloring options')
    p1 = g1.add_mutually_exclusive_group()
    p1.add_argument('--fill_color',
                    help='Can be either hexadecimal (ie. "#RRGGBB" '
                         'or 0xRRGGBB).')
    p1.add_argument('--dict_colors',
                    help='Dictionnary mapping basename to color.\n'
                         'Same convention as --fill_color.')
    p1.add_argument('--use_dps', metavar='DPS_KEY',
                    help='Use the data_per_streamline (scalar) for coloring, '
                         'linear from min to max.')
    p1.add_argument('--use_dpp', metavar='DPP_KEY',
                    help='Use the data_per_point (scalar) for coloring, '
                         'linear from min to max.')
    p.add_argument('--colormap', default='jet',
                   help='Select the colormap for colored trk (dps/dpp) '
                        '[%(default)s].')
    p.add_argument('--log', action='store_true',
                   help='Apply a base 10 logarithm for colored trk (dps/dpp).')

    g2 = p.add_argument_group(title='Output options')
    p2 = g2.add_mutually_exclusive_group()
    p2.add_argument('--out_suffix', default='colored',
                    help='Specify suffix to append to input basename.')
    p2.add_argument('--out_tractogram',
                    help='Output filename of colored tractogram (.trk).\n'
                         'Cannot be used with --dict_colors.')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    assert_inputs_exist(parser, args.in_tractograms)
    if args.out_suffix:
        if args.out_tractogram:
            args.out_suffix = ''
        else:
            args.out_suffix = '_'+args.out_suffix

    if len(args.in_tractograms) > 1 and args.out_tractogram:
        parser.error('Using multiple inputs, use --out_suffix.')
    out_filenames = []
    for filename in args.in_tractograms:
        base, ext = os.path.splitext(filename) if args.out_tractogram is None \
            else os.path.splitext(args.out_tractogram)
        if not ext == '.trk':
            logging.warning('Input is TCK file, will be converted to TRK.')
        out_filenames.append('{}{}{}'.format(base, args.out_suffix, '.trk'))
    assert_outputs_exist(parser, args, out_filenames)

    sft = load_tractogram_with_reference(parser, args, filename)
    cmap = plt.get_cmap(args.colormap)
    for i, filename in enumerate(args.in_tractograms):
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
        elif args.use_dps:
            data = np.squeeze(sft.data_per_streamline[args.use_dps])
            # I believe it works well for gaussian distribution, but
            # COMMIT has very weird outliers values
            if args.use_dps == 'commit_weights':
                data = np.clip(data, np.quantile(data, 0.05),
                               np.quantile(data, 0.95))
            if args.log:
                data[data > 0] = np.log10(data[data > 0])
            data -= np.min(data)
            data = data / np.max(data) if np.max(data) > 0 else data
            color = cmap(data)[:, 0:3] * 255
        elif args.use_dpp:
            data = np.squeeze(sft.data_per_point[args.use_dpp]._data)
            if args.log:
                data[data > 0] = np.log10(data[data > 0])
            data += np.min(data)
            data = data / np.max(data) if np.max(data) > 0 else data
            color = cmap(data)[:, 0:3] * 255

        if color is None:
            color = '0x000000'

        if isinstance(color, str):
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
        elif len(color) == len(sft):
            tmp = [np.tile([color[i][0], color[i][1], color[i][2]],
                           (len(sft.streamlines[i]), 1))
                   for i in range(len(sft.streamlines))]
            sft.data_per_point['color'] = tmp
        elif len(color) == len(sft.streamlines._data):
            sft.data_per_point['color'] = sft.streamlines
            sft.data_per_point['color']._data = color
        save_tractogram(sft, out_filename)


if __name__ == '__main__':
    main()
