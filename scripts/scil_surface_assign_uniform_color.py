#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import argparse
import json
import logging
import os

from dipy.io.surface import load_surface, save_surface
import numpy as np

from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg,
                             add_reference_arg,
                             add_surface_spatial_arg,
                             add_vtk_legacy_arg,
                             convert_stateful_str_to_enum)
from scilpy.viz.color import format_hexadecimal_color_to_rgb


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_surfaces', nargs='+',
                   help='Input surface(s) (VTK + PIAL + GII supported).')

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
    p2.add_argument('--out_surface', metavar='FILE',
                    help='Output filename of colored Surface (VTK supported).')

    add_surface_spatial_arg(p)
    add_vtk_legacy_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    if len(args.in_surfaces) > 1 and args.out_surface:
        parser.error('Using multiple inputs, use --out_suffix.')
    if args.dict_colors and args.out_surface:
        parser.error('Using --dict_colors, use --out_suffix.')

    assert_inputs_exist(parser, args.in_surfaces, args.reference)
    convert_stateful_str_to_enum(args)

    if args.reference is None:
        parser.error('A reference file is required to determine the space.\n'
                     'Please provide one using --reference')

    if args.out_suffix and args.out_suffix[0] != '_':
        args.out_suffix = '_' + args.out_suffix

    if args.out_surface:
        out_filenames = [args.out_surface]
        _, ext = os.path.splitext(args.out_surface)
    else:  # args.out_suffix
        out_filenames = []
        for filename in args.in_surfaces:
            base, ext = os.path.splitext(filename)
            out_filenames.append('{}{}{}'
                                 .format(base, args.out_suffix, ext))
    assert_outputs_exist(parser, args, out_filenames)

    # Loading (except tractograms, in loop)
    dict_colors = None
    if args.dict_colors:
        with open(args.dict_colors, 'r') as data:
            dict_colors = json.load(data)

    # Processing
    for i, filename in enumerate(args.in_surfaces):
        color = None

        sfs = load_surface(filename, args.reference,
                           from_space=args.source_space,
                           from_origin=args.source_origin)

        if args.dict_colors:
            base, ext = os.path.splitext(filename)
            base = os.path.basename(base)
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

        colors = np.tile([red, green, blue], (len(sfs.vertices), 1))

        sfs.data_per_point['RGB'] = colors
        save_surface(sfs, out_filenames[i],
                     to_space=args.destination_space,
                     to_origin=args.destination_origin,
                     legacy_vtk_format=args.legacy_vtk_format)


if __name__ == '__main__':
    main()
