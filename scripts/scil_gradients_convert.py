#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert gradient tables between FSL and MRtrix formats.

Formerly: scil_convert_gradients_mrtrix_to_fsl.py or
scil_convert_gradients_fsl_to_mrtrix.py
"""

import argparse
import logging

from scilpy.io.utils import (assert_gradients_filenames_valid,
                             assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg, add_verbose_arg)
from scilpy.io.gradients import fsl2mrtrix, mrtrix2fsl


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('gradients', nargs='+', metavar='GRADIENT_FILE(S)',
                   help='Path(s) to the gradient file(s). Either FSL '
                   '(.bval, .bvec) or MRtrix (.b).')

    p.add_argument('output', type=str,
                   help='Basename of output without extension. Extension(s) '
                        'will be added automatically (.b for MRtrix, '
                        '.bval/.bvec for FSL.')

    grad_format_group = p.add_mutually_exclusive_group(required=True)
    grad_format_group.add_argument('--input_fsl', action='store_true',
                                   help='FSL format.')
    grad_format_group.add_argument('--input_mrtrix', action='store_true',
                                   help='MRtrix format.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    input_is_fsl = args.input_fsl

    assert_gradients_filenames_valid(parser, args.gradients, input_is_fsl)
    assert_inputs_exist(parser, args.gradients)

    if input_is_fsl:
        output = args.output + '.b'
        assert_outputs_exist(parser, args, output)
        fsl_bval, fsl_bvec = args.gradients
        fsl2mrtrix(fsl_bval, fsl_bvec, args.output)
    else:
        output = [args.output + '.bval', args.output + '.bvec']
        assert_outputs_exist(parser, args, output)
        mrtrix_b = args.gradients[0]
        mrtrix2fsl(mrtrix_b, args.output)


if __name__ == "__main__":
    main()
