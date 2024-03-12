#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the RISH (Rotationally Invariant Spherical Harmonics) features of an SH
signal [1].

Each RISH feature map is the total energy of its associated order.
Mathematically, it is the sum of the squared SH coefficients of the SH order.

This script supports both symmetrical and asymmetrical SH images as input, of
any SH order.

Each RISH feature will be saved as a separate file.

[1] Mirzaalian, Hengameh, et al. "Harmonizing diffusion MRI data across
multiple sites and scanners." MICCAI 2015.
https://scholar.harvard.edu/files/hengameh/files/miccai2015.pdf

Formerly: scil_compute_rish_from_sh.py
"""
import argparse
import logging

from dipy.reconst.shm import order_from_ncoef, sph_harm_ind_list
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg,
                             assert_headers_compatible)
from scilpy.reconst.sh import compute_rish


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh',
                   help='Path of the sh image. They can be formatted in any '
                        'sh basis, but we \nexpect it to be a symmetrical '
                        'one. Else, provide --full_basis.')
    p.add_argument('out_prefix',
                   help='Prefix of the output RISH files to save. Suffixes '
                        'will be \nbased on the sh orders.')
    p.add_argument('--full_basis', action="store_true",
                   help="Input SH image uses a full SH basis (asymmetrical).")
    p.add_argument('--mask',
                   help='Path to a binary mask.\nOnly data inside the mask '
                        'will be used for computation.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_sh, optional=args.mask)
    assert_headers_compatible(parser, args.in_sh, optional=args.mask)

    # Load data
    sh_img = nib.load(args.in_sh)
    sh = sh_img.get_fdata(dtype=np.float32)
    mask = get_data_as_mask(nib.load(args.mask),
                            dtype=bool) if args.mask else None

    # Precompute output filenames to check if they exist
    sh_order = order_from_ncoef(sh.shape[-1], full_basis=args.full_basis)
    _, order_ids = sph_harm_ind_list(sh_order, full_basis=args.full_basis)
    orders = sorted(np.unique(order_ids))
    output_fnames = ["{}{}.nii.gz".format(args.out_prefix, i) for i in orders]
    assert_outputs_exist(parser, args, output_fnames)

    # Compute RISH features
    rish, final_orders = compute_rish(sh, mask, full_basis=args.full_basis)

    # Make sure the precomputed orders match the orders returned
    assert np.all(orders == np.array(final_orders))

    # Save each RISH feature as a separate file
    for i, fname in enumerate(output_fnames):
        logging.info("Saving {}".format(fname))
        nib.save(nib.Nifti1Image(rish[..., i], sh_img.affine), fname)


if __name__ == '__main__':
    main()
