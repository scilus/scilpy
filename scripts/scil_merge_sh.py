#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge a list of Spherical Harmonics files.

This merges the coefficients of multiple Spherical Harmonics files
by taking, for each coefficient, the one with the largest magnitude.

Can be used to merge fODFs computed from different shells into 1, while
conserving the most relevant information.

Based on [1].
"""

import argparse

import nibabel as nib
import numpy as np

from scilpy.io.image import assert_same_resolution
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


EPILOG = """
Reference:
[1] Garyfallidis, E., Zucchelli, M., Houde, J-C., Descoteaux, M.
    How to perform best ODF reconstruction from the Human Connectome
    Project sampling scheme?
    ISMRM 2014.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__, epilog=EPILOG)

    p.add_argument('in_shs', nargs="+",
                   help='List of SH files.')
    p.add_argument('out_sh',
                   help='output SH file.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_shs)
    assert_outputs_exist(parser, args, args.out_sh)
    assert_same_resolution(args.in_shs)

    first_im = nib.load(args.in_shs[0])
    out_coeffs = first_im.get_fdata(dtype=np.float32)

    for sh_file in args.in_shs[1:]:
        im = nib.load(sh_file)
        im_dat = im.get_fdata(dtype=np.float32)

        out_coeffs = np.where(np.abs(im_dat) > np.abs(out_coeffs),
                              im_dat, out_coeffs)

    nib.save(nib.Nifti1Image(out_coeffs, first_im.affine,
                             header=first_im.header),
             args.out_sh)


if __name__ == '__main__':
    main()
