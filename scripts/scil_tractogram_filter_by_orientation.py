#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to filter streamlines based on their distance traveled in a specific
dimension (x, y, or z).

Useful to help differentiate bundles.

Examples: In a brain aligned with x coordinates in left - right axis and y
coordinates in anterior-posterior axis, a streamline from the ...
    - corpus callosum will likely travel a very short distance in the y axis.
    - cingulum will likely travel a very short distance in the x axis.

Note: we consider that x, y, z are the coordinates of the streamlines; we
do not verify if they are aligned with the brain's orientation.

Formerly: scil_filter_streamlines_by_orientation.py
"""

import argparse
import json
import logging

import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference, \
    save_tractogram
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tractograms.streamline_operations import \
    filter_streamlines_by_total_length_per_dim


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__)

    p.add_argument('in_tractogram',
                   help='Streamlines input file name.')
    p.add_argument('out_tractogram',
                   help='Streamlines output file name.')

    p.add_argument('--min_x', default=0., type=float,
                   help='Minimum distance in the first dimension, in mm.'
                        '[%(default)s]')
    p.add_argument('--max_x', default=np.inf, type=float,
                   help='Maximum distance in the first dimension, in mm.'
                        '[%(default)s]')
    p.add_argument('--min_y', default=0., type=float,
                   help='Minimum distance in the second dimension, in mm.'
                        '[%(default)s]')
    p.add_argument('--max_y', default=np.inf, type=float,
                   help='Maximum distance in the second dimension, in mm.'
                        '[%(default)s]')
    p.add_argument('--min_z', default=0., type=float,
                   help='Minimum distance in the third dimension, in mm.'
                        '[%(default)s]')
    p.add_argument('--max_z', default=np.inf, type=float,
                   help='Maximum distance in the third dimension, in mm.'
                        '[%(default)s]')
    p.add_argument('--use_abs', action='store_true',
                   help="If set, will use the total of distances in absolute "
                        "value (ex, coming back on yourself will contribute "
                        "to the total distance instead of cancelling it).")

    p.add_argument('--no_empty', action='store_true',
                   help='Do not write file if there is no streamline.')
    p.add_argument('--display_counts', action='store_true',
                   help='Print streamline count before and after filtering.')
    p.add_argument('--save_rejected', metavar='filename',
                   help="Save the SFT of rejected streamlines.")

    add_json_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram, args.save_rejected)

    if args.min_x == 0 and np.isinf(args.max_x) and \
       args.min_y == 0 and np.isinf(args.max_y) and \
       args.min_z == 0 and np.isinf(args.max_z):
        logging.warning("You have not specified min or max in any direction. "
                        "Output will simply be a copy of your input!")

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    computed_rejected_sft = args.save_rejected is not None
    new_sft, indices, rejected_sft = \
        filter_streamlines_by_total_length_per_dim(
            sft, [args.min_x, args.max_x], [args.min_y, args.max_y],
            [args.min_z, args.max_z], args.use_abs, computed_rejected_sft)

    if args.display_counts:
        sc_bf = len(sft.streamlines)
        sc_af = len(new_sft.streamlines)
        print(json.dumps({'streamline_count_before_filtering': int(sc_bf),
                         'streamline_count_after_filtering': int(sc_af)},
                         indent=args.indent))

    save_tractogram(new_sft, args.out_tractogram,
                    args.no_empty)

    if computed_rejected_sft:
        save_tractogram(rejected_sft, args.save_rejected,
                        args.no_empty)


if __name__ == "__main__":
    main()
