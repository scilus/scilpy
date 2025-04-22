#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This script is used to alter a bundle while keeping it similar enough to the
original version (keeping the dice coefficient above a given threshold).
The script will subsample, trim, cut, replace or transform the streamlines
until the minimum dice is reached.

Various operations choices cannot be combined in one run, use the script
multiple times if needed.

All operations use a dichotomic search to find the parameter that gets as close
as possible to the specified minimum dice coefficient (with an epsilon for
convergence).

- The subsample operation will remove streamlines until the minimum dice is
  reached. This affects the whole bundle.

- The trim operation will use the lowest density voxels (starting at 1) to
  remove points from the streamlines until the minimum dice is reached.
  This typically affect the edge of the bundle.

- The cut operation will remove points from the start (or end) streamlines
  until the minimum dice is reached. This affects one end of the bundle.

- The replace operation will upsample the tractogram (generate new streamlines
  with noise) and then subsample the tractogram. This effectively replaces
  streamlines with similar ones until the minimum dice is reached.
  This affects the whole bundle.

- The transform operation will apply random rotations to the streamlines
  until the minimum dice is reached. This affects the whole bundle.
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tractograms.streamline_operations import \
    remove_overlapping_points_streamlines, \
    filter_streamlines_by_nb_points, cut_invalid_streamlines
from scilpy.tractograms.tractogram_operations import transform_streamlines_alter, \
    trim_streamlines_alter, cut_streamlines_alter, subsample_streamlines_alter, \
    replace_streamlines_alter, shuffle_streamlines, shuffle_streamlines_orientation
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_bundle',
                   help='Input bundle filename (.trk, .tck).')
    p.add_argument('out_bundle',
                   help='Output bundle filename (.trk, .tck).')

    p.add_argument('--min_dice', type=float, default=0.90,
                   help='Minimum dice to reach [%(default)s].')
    p.add_argument('--epsilon', type=float, default=0.01,
                   help='Epsilon for the convergence [%(default)s].')

    g = p.add_argument_group(title='Alteration options')
    g1 = g.add_mutually_exclusive_group(required=True,)
    g1.add_argument('--subsample', action='store_true',
                    help='Pick a subset of streamlines.')
    g1.add_argument('--trim', action='store_true',
                    help='Trim streamlines using low density voxel.')
    g1.add_argument('--cut', action='store_true',
                    help='Cut streamlines from endpoints.')
    g1.add_argument('--replace', action='store_true',
                    help='Replace streamlines with similar ones.')
    g1.add_argument('--transform', action='store_true',
                    help='Transform streamlines using a random linear '
                         'transformation.')
    g.add_argument('--from_end', action='store_true',
                   help='Cut streamlines from the tail rather than the head.\n'
                        'Only available with --cut.')
    g.add_argument('--save_transform', metavar='FILE',
                   help='Save the transformation matrix to a file.\n'
                        'Only available with --transform.')

    p.add_argument('--seed', '-s', type=int, default=None,
                   help='Seed for RNG. Default based on --min_dice.')
    p.add_argument('--shuffle', action='store_true',
                   help='Shuffle the streamlines and orientation after'
                        'alteration.')

    add_overwrite_arg(p)
    add_verbose_arg(p)
    add_reference_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundle, args.reference)
    assert_outputs_exist(parser, args, [args.out_bundle])
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if args.from_end and not args.cut:
        parser.error('The --from_end option is only available with --cut.')

    if args.save_transform and not args.transform:
        parser.error('The --save_transform option is only available with '
                     '--transform.')

    if args.seed is None:
        np.random.seed(int(args.min_dice * 1000))
    else:
        np.random.seed(args.seed)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)

    # Alter the streamlines with only one operation at a time
    if args.subsample:
        altered_sft = subsample_streamlines_alter(sft, args.min_dice,
                                                  epsilon=args.epsilon)
    elif args.trim:
        altered_sft = trim_streamlines_alter(sft, args.min_dice,
                                             epsilon=args.epsilon)
    elif args.cut:
        altered_sft = cut_streamlines_alter(sft, args.min_dice,
                                            epsilon=args.epsilon,
                                            from_end=args.from_end)
    elif args.replace:
        altered_sft = replace_streamlines_alter(sft, args.min_dice,
                                                epsilon=args.epsilon)
    elif args.transform:
        altered_sft, matrix = transform_streamlines_alter(sft, args.min_dice,
                                                          epsilon=args.epsilon)

    # Some operations could have generated invalid streamlines
    altered_sft, _ = cut_invalid_streamlines(altered_sft)
    altered_sft = filter_streamlines_by_nb_points(altered_sft, min_nb_points=2)
    altered_sft = remove_overlapping_points_streamlines(altered_sft)

    if args.shuffle:
        altered_sft = shuffle_streamlines(altered_sft)
        altered_sft = shuffle_streamlines_orientation(altered_sft)

    if args.save_transform:
        np.savetxt(args.save_transform, matrix)

    save_tractogram(altered_sft, args.out_bundle)


if __name__ == "__main__":
    main()
