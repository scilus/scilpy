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

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             add_reference_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.streamlines import cut_invalid_streamlines


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
                   help='Consider streamlines with overlapping points invalid.'
                   )
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

    # Equivalent of add_bbox_arg(p): always ignoring invalid streamlines for
    # this script.
    args.bbox_check = False

    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, args.out_tractogram)

    if args.threshold < 0:
        parser.error("Threshold must be positive.")

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    ori_len = len(sft)
    if args.cut_invalid:
        sft, cutting_counter = cut_invalid_streamlines(sft)
        logging.warning('Cut {} invalid streamlines.'.format(cutting_counter))
    else:
        sft.remove_invalid_streamlines()

    indices = []
    if args.remove_single_point:
        # Will try to do a PR in Dipy
        indices = [i for i in range(len(sft)) if len(sft.streamlines[i]) <= 1]

    if args.remove_overlapping_points:
        for i in np.setdiff1d(range(len(sft)), indices):
            norm = np.linalg.norm(np.diff(sft.streamlines[i], axis=0),
                                  axis=1)
            if (norm < args.threshold).any():
                indices.append(i)

    indices = np.setdiff1d(range(len(sft)), indices).astype(np.uint32)
    if len(indices):
        new_sft = sft[indices]
    else:
        new_sft = StatefulTractogram.from_sft([], sft)

    logging.warning('Removed {} invalid streamlines.'.format(
        ori_len - len(new_sft)))

    if len(new_sft) > 0 or (not args.no_empty and len(new_sft) == 0):
        save_tractogram(new_sft, args.out_tractogram)
    else:
        logging.warning('No valid streamline, not saving due to --no_empty.')


if __name__ == "__main__":
    main()
