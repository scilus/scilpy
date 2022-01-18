#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the RISH (Rotationally Invariant Spherical Harmonics) features
of an SH signal [1].

Each RISH feature map is the total energy of its
associated order. Mathematically, it is the sum of the squared SH
coefficients of the SH order.

This script supports symmetrical SH images as input, of any SH order.

[1] Mirzaalian, Hengameh, et al. "Harmonizing diffusion MRI data across
multiple sites and scanners." MICCAI 2015.
https://scholar.harvard.edu/files/hengameh/files/miccai2015.pdf
"""
import argparse

import nibabel as nib
import numpy as np

from scilpy.io.utils import add_overwrite_arg, assert_inputs_exist, \
    assert_outputs_exist
from scilpy.reconst.sh import compute_rish


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh',
                   help='Path of the sh image.')
    p.add_argument('out_rish',
                   help='Name of the output RISH file to save.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_sh])
    assert_outputs_exist(parser, args, args.out_rish)

    sh = nib.load(args.in_sh)
    rish = compute_rish(sh)

    nib.save(nib.Nifti1Image(rish.astype(np.float32), sh.affine), args.out_rish)


if __name__ == '__main__':
    main()
