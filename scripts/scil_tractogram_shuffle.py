#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shuffle the ordering of streamlines.

Formerly: scil_shuffle_streamlines.py
"""

import argparse
import logging

from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist, add_verbose_arg,
                             assert_outputs_exist)
from scilpy.tractograms.tractogram_operations import shuffle_streamlines


def _build_arg_parser():

    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Input tractography file.')
    p.add_argument('out_tractogram',
                   help='Output tractography file.')
    p.add_argument('--seed', type=int, default=None,
                   help='Random number generator seed [%(default)s].')

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
    shuffled_sft = shuffle_streamlines(sft, rng_seed=args.seed)
    save_tractogram(shuffled_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
