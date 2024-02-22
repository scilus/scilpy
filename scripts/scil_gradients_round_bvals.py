#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select b-values on specific b-value shells.

With the --tolerance argument, this is useful for sampling schemes where
b-values of a shell are not all identical. Adjust the tolerance to vary the
accepted interval around the targetted b-value.

For example, a b-value of 2000 and a tolerance of 20 will select all b-values
between [1980, 2020] and round them to the value of 2000.

>> scil_gradients_round_bvals.py bvals 0 1000 2000 newbvals --tolerance 20

Formerly: scil_resample_bvals.py
"""

import argparse
import logging

from dipy.io import read_bvals_bvecs
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.gradients.bvec_bval_tools import round_bvals_to_shell


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('in_bval',
                        help='The b-values in FSL format.')
    parser.add_argument(
        'shells', nargs='+', type=int,
        help='The list of expected shells. For example 0 1000 2000.\n'
             'All b-values in the b_val file should correspond to one given '
             'shell (up to the tolerance).')
    parser.add_argument('out_bval',
                        help='The name of the output b-values.')
    parser.add_argument(
        'tolerance', type=int,
        help='The tolerated gap between the b-values to extract and the \n'
             'actual b-values. Expecting an integer value. Comparison is \n'
             'strict: a b-value of 1010 with a tolerance of 10 is NOT \n'
             'included in shell 1000. Suggestion: 20.')

    add_verbose_arg(parser)
    add_overwrite_arg(parser)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bval)
    assert_outputs_exist(parser, args, args.out_bval)

    bvals, _ = read_bvals_bvecs(args.in_bval, None)

    new_bvals = round_bvals_to_shell(bvals, args.shells, tol=args.tolerance)

    logging.info("new bvals: {}".format(new_bvals))
    new_bvals.reshape((1, len(new_bvals)))
    np.savetxt(args.out_bval, new_bvals, '%d')


if __name__ == "__main__":
    main()
