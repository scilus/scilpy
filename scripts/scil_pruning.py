#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_reference)
from scilpy.tracking.tools import filter_streamlines_by_length


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Keep only streamlines between [min, max] length',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_bundle',
                   help='Bundle to prune.')

    add_reference(p)

    p.add_argument('out_bundle',
                   help='Pruned bundle.')

    p.add_argument('--min_length',
                   default=20., type=float,
                   help='Keep streamlines longer than min_length.' +
                        '[%(default)s]')
    p.add_argument('--max_length',
                   default=200., type=float,
                   help='Keep streamlines shorter than max_length. ' +
                        '[%(default)s]')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.bundle)
    assert_outputs_exist(parser, args, args.pruned_bundle)

    if args.min_length < 0:
        parser.error('--min_length {} should be at least 0'
                     .format(args.min_length))
    if args.max_length <= args.min_length:
        parser.error('--max_length {} should be greater than --min_length'
                     .format(args.max_length))

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)

    pruned_streamlines, _, _ = filter_streamlines_by_length(sft.streamlines,
                                                            args.min_length,
                                                            args.max_length)

    if not pruned_streamlines:
        print("Pruning removed all the streamlines. Please adjust "
              "--{min,max}_length")
    else:
        sft = StatefulTractogram(pruned_streamlines, sft, Space.RASMM)
        save_tractogram(sft, args.out_bundle)


if __name__ == '__main__':
    main()
