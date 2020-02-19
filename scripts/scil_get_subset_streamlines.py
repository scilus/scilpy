#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from dipy.io.streamline import save_tractogram
from scilpy.io.streamlines import load_tractogram_with_reference

from scilpy.tracking.tools import get_subset_streamlines
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_reference_arg)


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Get a subset of streamlines.')
    p.add_argument('in_tractogram',
                   help='Streamlines input file name.')
    p.add_argument('max_num_streamlines', type=int,
                   help='Maximum number of streamlines to output.')
    p.add_argument('out_tractogram',
                   help='Streamlines output file name.')
    p.add_argument('--seed', default=None, type=int,
                   help='Use a specific random seed for the resampling.')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():

    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    new_sft = get_subset_streamlines(sft, args.max_num_streamlines, args.seed)

    save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
