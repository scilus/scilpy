#! /usr/bin/env python

import argparse

from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exists)
from scilpy.utils.flip_tools import flip_mrtrix_encoding_scheme, flip_fsl_bvecs
from scilpy.utils.util import str_to_index


def build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description='Flip one or more axes of the '
                                            ' encoding scheme matrix.')

    p.add_argument('encoding_file', action='store', metavar='encoding_file',
                   type=str,
                   help='Path to encoding file.')

    p.add_argument('flipped_encoding', action='store',
                   metavar='flipped_encoding', type=str,
                   help='Path to the flipped encoding file.')

    p.add_argument('axes', action='store', metavar='dimension',
                   choices=['x', 'y', 'z'], nargs='+',
                   help='The axes you want to flip. eg: to flip the x '
                        ' and y axes use: x y')

    gradients_type = p.add_mutually_exclusive_group(required=True)
    gradients_type.add_argument('--fsl', dest='fsl_bvecs', action='store_true',
                                help='Specify fsl format')
    gradients_type.add_argument('--mrtrix', dest='fsl_bvecs',
                                action='store_false',
                                help='Specify mrtrix format')

    p.add_argument('-f', action='store_true', dest='overwrite',
                   help='Force (overwrite output file). [%(default)s]')

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.encoding_file])
    assert_outputs_exists(parser, args, [args.flipped_encoding])

    indices = [str_to_index(axis) for axis in list(args.axes)]
    if args.fsl_bvecs:
        flip_fsl_bvecs(args.encoding_file, args.flipped_encoding, indices)
    else:
        flip_mrtrix_encoding_scheme(args.encoding_file, args.flipped_encoding,
                                    indices)


if __name__ == "__main__":
    main()
