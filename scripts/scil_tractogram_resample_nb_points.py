#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to resample a set of streamlines to either a new number of points per
streamline or to a fixed step size. WARNING: data_per_point is not carried.

Formerly: scil_resample_streamlines.py
"""
import argparse
import logging

from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tractograms.streamline_operations import \
    resample_streamlines_num_points, resample_streamlines_step_size


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__)

    p.add_argument('in_tractogram',
                   help='Streamlines input file name.')
    p.add_argument('out_tractogram',
                   help='Streamlines output file name.')

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--nb_pts_per_streamline', type=int,
                   help='Number of points per streamline in the output.')
    g.add_argument('--step_size', type=float,
                   help='Step size in the output (in mm).')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    if args.nb_pts_per_streamline:
        new_sft = resample_streamlines_num_points(sft,
                                                  args.nb_pts_per_streamline)
    else:
        new_sft = resample_streamlines_step_size(sft, args.step_size)

    save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
