#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from dipy.io.streamline import save_tractogram

from scilpy.tracking.tools import (resample_streamlines_num_points,
                                   resample_streamlines_step_size)
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_reference_arg)
from scilpy.io.streamlines import load_tractogram_with_reference


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Resample a set of streamlines to either a new number of '
                    'points per streamline or to a fixed step size.\n'
                    'WARNING: data_per_point is not carried')
    p.add_argument('in_tractogram',
                   help='Streamlines input file name.')
    p.add_argument('out_tractogram',
                   help='Streamlines output file name.')

    g = p.add_mutually_exclusive_group()
    g.add_argument('--nb_pts_per_streamline', type=int,
                   help='Number of points per streamline in the output.')
    g.add_argument('--step_size', type=float,
                   help='Step size in the output (in mm).')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():

    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    if args.nb_pts_per_streamline:
        new_sft = resample_streamlines_num_points(sft,
                                                  args.nb_pts_per_streamline)
    elif args.step_size:
        new_sft = resample_streamlines_step_size(sft, args.step_size)
    else:
        raise ValueError("Either nb_points_per_streamline or step_size should"
                         "be defined.")

    save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
