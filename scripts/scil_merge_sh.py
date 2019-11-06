#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge 2 Spherical Harmonics files.

This merges the coefficients of 2 Spherical Harmonics files
by taking, for each coefficient, the one with the largest magnitude.

Can be used to merge fODFs computed from 2 different shells into 1, while
conserving the most relevant information.

Based on [1].
"""


EPILOG = """
Reference:
    [1] Garyfallidis, E., Zucchelli, M., Houde, J-C., Descoteaux, M.
        How to perform best ODF reconstruction from the Human Connectome
        Project sampling scheme?
        ISMRM 2014.
"""

import argparse

import nibabel as nb
import numpy as np

from scilpy.io.image import assert_same_resolution
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=EPILOG,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('first_in',
                        help='first input SH file')
    parser.add_argument('second_in',
                        help='second input SH file')
    parser.add_argument('out_sh',
                        help='output SH file')

    add_overwrite_arg(parser)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.first_in, args.second_in])
    assert_outputs_exist(parser, args, [args.out_sh])

    assert_same_resolution(args.first_in, args.second_in)

    first_im = nb.load(args.first_in)
    second_im = nb.load(args.second_in)

    first_dat = first_im.get_data()
    second_dat = second_im.get_data()

    out_coeffs = np.where(np.abs(first_dat) > np.abs(second_dat),
                          first_dat, second_dat)
    nb.save(nb.Nifti1Image(out_coeffs, first_im.affine, first_im.header),
            args.out_sh)


if __name__ == '__main__':
    main()
