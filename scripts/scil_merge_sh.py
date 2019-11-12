#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import nibabel as nb
import numpy as np

from scilpy.io.image import assert_same_resolution
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)

"""
Merge a list of Spherical Harmonics files.

This merges the coefficients of multiple Spherical Harmonics files
by taking, for each coefficient, the one with the largest magnitude.

Can be used to merge fODFs computed from different shells into 1, while
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


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=EPILOG,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('sh_files', nargs="+",
                        help='List of SH files.')
    parser.add_argument('out_sh',
                        help='output SH file.')

    add_overwrite_arg(parser)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.sh_files)
    assert_outputs_exist(parser, args, args.out_sh)
    assert_same_resolution(args.sh_files)

    first_im = nb.load(args.sh_files[0])
    out_coeffs = first_im.get_data()

    for sh_file in args.sh_files[1:]:
        im = nb.load(sh_file)
        im_dat = im.get_data()

        out_coeffs = np.where(np.abs(im_dat) > np.abs(out_coeffs),
                              im_dat, out_coeffs)

    nb.save(nb.Nifti1Image(out_coeffs, first_im.affine, first_im.header),
            args.out_sh)


if __name__ == '__main__':
    main()
