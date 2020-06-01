#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flip one or more axes of the gradient sampling matrix. It will be saved in
the same format as input gradient sampling file.
"""
import argparse
import os

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.bvec_bval_tools import (flip_mrtrix_gradient_sampling,
                                          flip_fsl_gradient_sampling)
from scilpy.utils.util import str_to_index


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('gradient_sampling_file',
                   help='Path to gradient sampling file. (.bvec or .b)')

    p.add_argument('flipped_sampling_file',
                   help='Path to the flipped gradient sampling file.')

    p.add_argument('axes', metavar='dimension',
                   choices=['x', 'y', 'z'], nargs='+',
                   help='The axes you want to flip. eg: to flip the x '
                        'and y axes use: x y.')

    gradients_type = p.add_mutually_exclusive_group(required=True)
    gradients_type.add_argument('--fsl', dest='fsl_bvecs',
                                action='store_true',
                                help='Specify fsl format.')
    gradients_type.add_argument('--mrtrix', dest='fsl_bvecs',
                                action='store_false',
                                help='Specify mrtrix format.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.gradient_sampling_file)
    assert_outputs_exist(parser, args, args.flipped_sampling_file)

    indices = [str_to_index(axis) for axis in list(args.axes)]

    _, ext = os.path.splitext(args.gradient_sampling_file)

    if args.fsl_bvecs:
        if ext == '.bvec':
            flip_fsl_gradient_sampling(args.gradient_sampling_file,
                                       args.flipped_sampling_file,
                                       indices)
        else:
            parser.error('Extension for FSL format should be .bvec.')

    elif ext == '.b':
        flip_mrtrix_gradient_sampling(args.gradient_sampling_file,
                                      args.flipped_sampling_file,
                                      indices)
    else:
        parser.error('Extension for MRtrix format should .b.')


if __name__ == "__main__":
    main()
