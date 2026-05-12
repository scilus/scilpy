#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a binary mask based on a global SF threshold.
The script masks voxels where the max SF amplitude is below
either a relative factor or an absolute threshold.

The input can be either SH coefficients or peaks.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from dipy.data import get_sphere
from scilpy.io.stateful_image import StatefulImage
from scilpy.io.utils import (add_sh_basis_args, add_sphere_arg,
                             add_verbose_arg, add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             parse_sh_basis_arg)
from scilpy.reconst.utils import compute_sh_threshold_mask
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
    thr_g.add_argument('--factor', type=float,
                       help='Global SF threshold factor (0-1).')
    thr_g.add_argument('--absolute', type=float,
                       help='Global SF absolute threshold.')
    add_sh_basis_args(p)
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
                               sh_basis=sh_basis)
    from scilpy.reconst.utils import is_data_peaks
    print("--- is_data_peaks(simg.data):", is_data_peaks(simg.get_fdata()))
    data = simg.to_voxel_direction(sh_basis=sh_basis).astype(np.float32)
    print("--- is_data_peaks(simg.data):", is_data_peaks(simg.get_fdata()))
    print("--- is_data_peaks(data):", is_data_peaks(data))

    logging.info("Computing global SH threshold mask.")
    mask, global_max, threshold = compute_sh_threshold_mask(
        data, relative_factor=args.factor,
        absolute_threshold=args.absolute)

    logging.info("Global energy sum for SH: {:.4f}".format(global_max))
    if args.factor is not None:
        logging.info("Computed threshold: {:.4f} (Factor: {})".format(threshold,
                                                                     args.factor))
    else:
        logging.info("Absolute threshold used: {:.4f}".format(args.absolute))

    logging.info("Number of voxels in mask: {}".format(np.sum(mask)))

    # Save mask
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), simg.affine,
                               simg.header)
    nib.save(mask_img, args.out_mask)


if __name__ == "__main__":
    main()
