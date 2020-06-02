#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shuffle the ordering of streamlines.
"""

import argparse
import random

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

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
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    indices = np.arange(len(sft.streamlines))
    random.shuffle(indices, random=args.seed)

    streamlines = sft.streamlines[indices]
    data_per_streamline = sft.data_per_streamline[indices]
    data_per_point = sft.data_per_point[indices]

    shuffled_sft = StatefulTractogram.from_sft(streamlines, sft,
                                               data_per_streamline=data_per_streamline,
                                               data_per_point=data_per_point)
    save_tractogram(shuffled_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
