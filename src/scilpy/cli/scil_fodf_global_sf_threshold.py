#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a binary mask based on a global SF threshold.
The script masks voxels where the max SF amplitude is below
either a relative factor or an absolute threshold.

The absolute threshold can be estimated from the mean/median maximum fODF in the
ventricles, computed with scil_fodf_max_in_ventricles.

The input can be either SH coefficients or peaks. However, the vectors
cannot be normalized, as the amplitude is used for thresholding.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.stateful_image import StatefulImage
from scilpy.io.utils import (add_sh_basis_args, add_sphere_arg,
                             add_verbose_arg, add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             parse_sh_basis_arg)
from scilpy.reconst.utils import compute_sf_threshold_mask
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_odf',
                   help='Input ODF file (SH or Peaks) (.nii.gz).')
    p.add_argument('out_mask',
                   help='Output binary mask (.nii.gz).')

    thr_g = p.add_mutually_exclusive_group(required=True)
    thr_g.add_argument('--relative', type=float,
                       help='Global SF threshold relative factor (0-1).')
    thr_g.add_argument('--absolute', type=float,
                       help='Global SF absolute threshold.')
    add_sh_basis_args(p)
    add_sphere_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_odf)
    assert_outputs_exist(parser, args, args.out_mask)

    sh_basis, is_legacy = parse_sh_basis_arg(args)

    logging.info("Loading ODF data.")
    simg = StatefulImage.load(args.in_odf, is_orientation=True,
                               sh_basis=sh_basis, is_legacy=is_legacy)

    data = simg.to_voxel_direction(sh_basis=sh_basis,
                                   is_legacy=is_legacy).astype(np.float32)

    logging.info("Computing global SF threshold mask.")
    mask, global_max, threshold = compute_sf_threshold_mask(
        data, sphere_name=args.sphere, relative_factor=args.relative,
        absolute_threshold=args.absolute, sh_basis=sh_basis,
        is_legacy=is_legacy)

    logging.info("Global max SF amplitude: {:.4f}".format(global_max))
    if args.relative is not None:
        logging.info("Relative threshold: {:.4f} (Factor: {})".format(threshold,
                                                                     args.relative))
    else:
        logging.info("Absolute threshold used: {:.4f}".format(args.absolute))

    logging.info("Number of voxels in mask: {}".format(np.sum(mask)))

    # Save mask
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), simg.affine,
                               simg.header)
    nib.save(mask_img, args.out_mask)


if __name__ == "__main__":
    main()
