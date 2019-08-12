#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import logging
import os

import tractconverter as tc

from scilpy.tractanalysis.features import remove_loops_and_sharp_turns


DESCRIPTION = """
This script can be used to remove loops in two types of streamline datasets:

  - Whole brain: For this type, the script removes streamlines if they
    make a loop with an angle of more than 360 degrees. It's possible to change
    this angle with the -a option. Warning: Don't use --qb option for a
    whole brain tractography.

  - Bundle dataset: For this type, it is possible to remove loops and fibers
    outsides of bundle. For the sharp angle turn, use --qb option.

-------------------------------------------------------------------------------------
Reference:
QuickBundles based on [Garyfallidis12] Frontiers in Neuroscience, vol 6, no 175, 2012.
--------------------------------------------------------------------------------------
"""


def buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)
    p.add_argument('input', action='store',  metavar='input', type=str,
                   help='Tractogram input file name.')
    p.add_argument('output_clean', action='store',
                   metavar='output_clean', type=str,
                   help='Output tractogram without loops.')
    p.add_argument('--output_loops', action='store', type=str,
                   help='If set, saves detected looping streamlines')
    p.add_argument('--qb', dest='QB', action='store_true',
                   help='If set, uses QuickBundles to detect\n' +
                        'outliers (loops, sharp angle turns).\n' +
                        'Should mainly be used with bundles. '
                        '[%(default)s]')
    p.add_argument('--threshold', dest='threshold', action='store',
                   default=8., type=float,
                   help='Maximal streamline to bundle distance\n' +
                        'for a streamline to be considered as\n' +
                        'a tracking error. [%(default)s]')
    p.add_argument('-a', dest='angle', action='store',
                   default=360, type=float,
                   help='Maximum looping (or turning) angle of\n' +
                        'a streamline in degrees. [%(default)s]')
    p.add_argument('-f', action='store_true', dest='isForce',
                   help='Force (overwrite output file). [%(default)s]')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    in_filename = args.input
    out_filename_loops = args.output_loops
    out_filename_clean = args.output_clean
    streamlines_c = []
    loops = []

    if not tc.is_supported(args.input):
        parser.error('Input file must be one of {0}!'
                     .format(",".join(tc.FORMATS.keys())))

    in_format = tc.detect_format(in_filename)
    out_format_clean = tc.detect_format(out_filename_clean)

    if not in_format == out_format_clean:
            parser.error('Input and output must be of the same types!'
                         .format(",".join(tc.FORMATS.keys())))

    if args.output_loops:
        out_format_loops = tc.detect_format(out_filename_loops)
        if not in_format == out_format_loops:
            parser.error('Input and output loops must be of the same types!'
                         .format(",".join(tc.FORMATS.keys())))
        if os.path.isfile(args.output_loops):
            if args.isForce:
                logging.info('Overwriting "{0}".'.format(out_filename_loops))
            else:
                parser.error('"{0}" already exist! Use -f to overwrite it.'
                             .format(out_filename_loops))

    if os.path.isfile(args.output_clean):
        if args.isForce:
            logging.info('Overwriting "{0}".'.format(out_filename_clean))
        else:
            parser.error('"{0}" already exist! Use -f to overwrite it.'
                         .format(out_filename_clean))

    if args.threshold <= 0:
        parser.error('"{0}"'.format(args.threshold) + 'must be greater than 0')

    if args.angle <= 0:
        parser.error('"{0}"'.format(args.angle) + 'must be greater than 0')

    tract = in_format(in_filename)
    streamlines = [i for i in tract]

    if len(streamlines) > 1:
        streamlines_c, loops = remove_loops_and_sharp_turns(streamlines,
                                                            args.QB,
                                                            args.angle,
                                                            args.threshold)
    else:
        parser.error('Zero or one streamline in ' + '{0}'.format(in_filename) +
                     '. The file must have more than one streamline.')

    hdr = tract.hdr.copy()
    nb_points_init = hdr[tc.formats.header.Header.NB_POINTS]
    nb_points_clean = 0

    if len(streamlines_c) > 0:
        hdr[tc.formats.header.Header.NB_FIBERS] = len(streamlines_c)
        if in_format is tc.formats.vtk.VTK:
            for s in streamlines_c:
                nb_points_clean += len(s)
            hdr[tc.formats.header.Header.NB_POINTS] = nb_points_clean
        output_clean = out_format_clean.create(out_filename_clean, hdr)
        output_clean += streamlines_c
        output_clean.close()
    else:
        logging.warning("No clean streamlines in {0}".format(args.input))

    if len(loops) == 0:
        logging.warning("No loops in {0}".format(args.input))

    if args.output_loops and len(loops) > 0:
        hdr[tc.formats.header.Header.NB_FIBERS] = len(loops)
        if in_format is tc.formats.vtk.VTK:
            hdr[tc.formats.header.Header.NB_POINTS] = nb_points_init - nb_points_clean
        output_loops = out_format_loops.create(out_filename_loops, hdr)
        output_loops += loops
        output_loops.close()


if __name__ == "__main__":
    main()
