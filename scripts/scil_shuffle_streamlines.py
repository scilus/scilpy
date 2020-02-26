#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_args_parser():

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Shuffle the ordering of streamlines.')

    p.add_argument('in_tractogram',
                   help='Input tractography file.')
    p.add_argument('out_tractogram',
                   help='Output tractography file.')
    p.add_argument('--seed', type=int, default=None,
                   help='Random number generator seed [%(default)s].')
    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    streamlines = list(sft.get_streamlines_copy())
    random.shuffle(streamlines, random=args.seed)

    smoothed_sft = StatefulTractogram(streamlines, sft, Space.RASMM)
    save_tractogram(smoothed_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
