#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be used to remove loops in two types of streamline datasets:

  - Whole brain: For this type, the script removes streamlines if they
    make a loop with an angle of more than 360 degrees. It's possible to change
    this angle with the --angle option. Warning: Don't use --qb option for a
    whole brain tractography.

  - Bundle dataset: For this type, it is possible to remove loops and
    streamlines outside the bundle. For the sharp angle turn, use --qb option.

Formerly: scil_detect_streamlines_loops.py
"""

import argparse
import json
import logging

import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference, \
    save_tractogram
from scilpy.io.utils import (add_json_args,
                             add_verbose_arg,
                             add_overwrite_arg,
                             add_processes_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             check_tracts_same_format,
                             validate_nbr_processes, ranged_type)
from scilpy.tractograms.streamline_operations import \
    remove_loops_and_sharp_turns


EPILOG = """
References:
    QuickBundles, based on [Garyfallidis12] Frontiers in Neuroscience, 2012.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__, epilog=EPILOG)
    p.add_argument('in_tractogram',
                   help='Tractogram input file name.')
    p.add_argument('out_tractogram',
                   help='Output tractogram without loops.')
    p.add_argument('--looping_tractogram', metavar='out_filename',
                   help='If set, saves detected looping streamlines.')
    p.add_argument('--qb', nargs='?', metavar='threshold', dest='qb_threshold',
                   const=8., type=ranged_type(float, 0.0, None),
                   help='If set, uses QuickBundles to detect outliers (loops, '
                        'sharp angle \nturns). Given threshold is the maximal '
                        'streamline to bundle \ndistance for a streamline to '
                        'be considered as a tracking error.\nDefault if '
                        'set: [%(const)s]')
    p.add_argument('--angle', default=360, type=ranged_type(float, 0.0, 360.0),
                   help='Maximum looping (or turning) angle of\n' +
                        'a streamline in degrees. [%(default)s]')
    p.add_argument('--display_counts', action='store_true',
                   help='Print streamline count before and after filtering')
    p.add_argument('--no_empty', action='store_true',
                   help="If set, will not save outputs if they are empty.")

    add_json_args(p)
    add_processes_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram,
                         optional=args.looping_tractogram)
    check_tracts_same_format(parser, [args.in_tractogram, args.out_tractogram,
                                      args.looping_tractogram])
    nbr_cpu = validate_nbr_processes(parser, args)

    # Loading
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    nb_streamlines = len(sft.streamlines)

    if nb_streamlines < 1:
        parser.error(
            'Zero or one streamline in {}. The file must have more than one '
            'streamline.'.format(args.in_tractogram))

    # Processing
    ids_clean = remove_loops_and_sharp_turns(
        sft.streamlines, args.angle, qb_threshold=args.qb_threshold,
        num_processes=nbr_cpu)
    if len(ids_clean) == 0:
        logging.warning('No clean streamlines in {}. They are all looping '
                        'streamlines? Check your parameters.'
                        .format(args.in_tractogram))
    sft_clean = sft[ids_clean]

    if args.display_counts:
        sc_af = len(sft_clean.streamlines)
        print(json.dumps({'streamline_count_before_filtering': nb_streamlines,
                         'streamline_count_after_filtering': int(sc_af)},
                         indent=args.indent))

    # Saving
    save_tractogram(sft_clean, args.out_tractogram,
                    args.no_empty)
    if args.looping_tractogram:
        ids_removed = np.setdiff1d(np.arange(nb_streamlines), ids_clean)
        sft_l = sft[ids_removed]
        save_tractogram(sft_l, args.looping_tractogram,
                        args.no_empty)


if __name__ == "__main__":
    main()
