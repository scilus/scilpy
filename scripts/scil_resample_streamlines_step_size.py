#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from dipy.io.streamline import save_tractogram

from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_reference_arg)
from scilpy.io.streamlines import load_tractogram_with_reference


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Resample a set of streamlines to a fixed step size.\n'
                    'WARNING: data_per_point is not carried')
    p.add_argument('in_tractogram',
                   help='Streamlines input file name.')
    p.add_argument('step_size', type=int,
                   help='Step size in the output (in mm).')
    p.add_argument('out_tractogram',
                   help='Streamlines output file name.')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():

    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    new_sft = resample_streamlines_step_size(sft, args.step_size)

    save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
