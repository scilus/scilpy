#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Removal of streamlines that are out of the volume bounding box. In voxel space
no negative coordinate and no above volume dimension coordinate are possible.
Any streamline that do not respect these two conditions are removed.

The --cut_invalid option will cut streamlines so that their longest segment are
within the bounding box

Formerly: scil_remove_invalid_streamlines.py
"""

import argparse
import logging

from scilpy.io.streamlines import load_tractogram_with_reference, \
    save_tractogram
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             add_reference_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.tractograms.streamline_operations import (
    cut_invalid_streamlines,
    remove_overlapping_points_streamlines,
    remove_single_point_streamlines)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy.')
    p.add_argument('out_tractogram',
                   help='Output filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy.')

    p.add_argument('--cut_invalid', action='store_true',
                   help='Cut invalid streamlines rather than removing them.\n'
                        'Keep the longest segment only.')
    p.add_argument('--remove_single_point', action='store_true',
                   help='Consider single point streamlines invalid.')
    p.add_argument('--remove_overlapping_points', action='store_true',
                   help='Consider streamlines with overlapping points invalid.')
    p.add_argument('--threshold', type=float, default=0.001,
                   help='Maximum distance between two points to be considered'
                        ' overlapping [%(default)s mm].')
    p.add_argument('--no_empty', action='store_true',
                   help='Do not save empty tractogram.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Equivalent of add_bbox_arg(p): always ignoring invalid streamlines for
    # this script.
    args.bbox_check = False

    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram)

    if args.threshold < 0:
        parser.error("Threshold must be positive.")

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    ori_len = len(sft)
    ori_len_pts = len(sft.streamlines._data)
    if args.cut_invalid:
        sft, cutting_counter = cut_invalid_streamlines(sft,
                                                       epsilon=args.threshold)
        logging.warning('Cut {} invalid streamlines.'.format(cutting_counter))
    else:
        sft.remove_invalid_streamlines()

    if args.remove_single_point:
        sft = remove_single_point_streamlines(sft)

    if args.remove_overlapping_points:
        sft = remove_overlapping_points_streamlines(sft, args.threshold)
        logging.warning("data_per_point will be discarded.")
        logging.warning('Removed {} overlapping points from tractogram.'.format(
            ori_len_pts - len(sft.streamlines._data)))

    logging.warning('Removed {} invalid streamlines.'.format(
        ori_len - len(sft)))

    save_tractogram(sft, args.out_tractogram, args.no_empty)


if __name__ == "__main__":
    main()
