#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generation of priors and enhanced-FOD from an example/template bundle.
The bundle must have been cleaned thorougly before use. The E-FOD can then
be used for bundle-specific tractography, but not for FOD metrics.

Formerly: scil_generate_priors_from_bundle.py
"""

import argparse
import logging
import os

from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_sh_basis_args,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             parse_sh_basis_arg,
                             assert_headers_compatible)
from scilpy.reconst.utils import find_order_from_nb_coeff
from scilpy.tractanalysis.todi import TrackOrientationDensityImaging


EPILOG = """
    References:
        [1] Rheault, Francois, et al. "Bundle-specific tractography with
        incorporated anatomical and orientational priors."
        NeuroImage 186 (2019): 382-398
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__, epilog=EPILOG,)
    p.add_argument('in_bundle',
                   help='Input bundle filename.')

    p.add_argument('in_fodf',
                   help='Input FOD filename.')

    p.add_argument('in_mask',
                   help='Mask to constrain the TODI spatial smoothing,\n'
                        'for example a WM mask.')
    add_sh_basis_args(p)
    p.add_argument('--todi_sigma', choices=[0, 1, 2, 3, 4],
                   default=1, type=int,
                   help='Smooth the orientation histogram.')
    p.add_argument('--sf_threshold', default=0.2, type=float,
                   help='Relative threshold for sf masking (0.0-1.0).')
    p.add_argument('--out_prefix', default='',
                   help='Add a prefix to all output filename, \n'
                        'default is no prefix.')
    p.add_argument('--out_dir', default='./',
                   help='Output directory for all generated files,\n'
                        'default is current directory.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    required = [args.in_bundle, args.in_fodf, args.in_mask]
    assert_inputs_exist(parser, required, args.reference)
    assert_headers_compatible(parser, required, reference=args.reference)

    out_efod = os.path.join(args.out_dir,
                            '{0}efod.nii.gz'.format(args.out_prefix))
    out_priors = os.path.join(args.out_dir,
                              '{0}priors.nii.gz'.format(args.out_prefix))
    out_todi_mask = os.path.join(args.out_dir,
                                 '{0}todi_mask.nii.gz'.format(args.out_prefix))
    out_endpoints_mask = os.path.join(args.out_dir,
                                      '{0}endpoints_mask.nii.gz'.format(
                                          args.out_prefix))

    if args.out_dir and not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    required = [out_efod, out_priors, out_todi_mask, out_endpoints_mask]
    assert_outputs_exist(parser, args, required)

    img_sh = nib.load(args.in_fodf)
    sh_shape = img_sh.shape
    sh_order = find_order_from_nb_coeff(sh_shape)
    sh_basis, is_legacy = parse_sh_basis_arg(args)
    img_mask = nib.load(args.in_mask)
    mask_data = get_data_as_mask(img_mask)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    if len(sft.streamlines) < 1:
        raise ValueError('The input bundle contains no streamline.')

    # Compute TODI from streamlines
    with TrackOrientationDensityImaging(img_mask.shape,
                                        'repulsion724') as todi_obj:
        todi_obj.compute_todi(sft.streamlines, length_weights=True)
        todi_obj.smooth_todi_dir()
        todi_obj.smooth_todi_spatial(sigma=args.todi_sigma)

        # Fancy masking of 1d indices to limit spatial dilation to WM
        sub_mask_3d = np.logical_and(
            mask_data, todi_obj.reshape_to_3d(todi_obj.get_mask()))
        sub_mask_1d = sub_mask_3d.flatten()[todi_obj.get_mask()]
        todi_sf = todi_obj.get_todi()[sub_mask_1d] ** 2

    # The priors should always be between 0 and 1
    # A minimum threshold is set to prevent misaligned FOD from disappearing
    todi_sf /= np.max(todi_sf, axis=-1, keepdims=True)
    todi_sf[todi_sf < args.sf_threshold] = args.sf_threshold

    # Memory friendly saving, as soon as possible saving then delete
    priors_3d = np.zeros(sh_shape)
    sphere = get_sphere('repulsion724')
    priors_3d[sub_mask_3d] = sf_to_sh(todi_sf, sphere,
                                      sh_order_max=sh_order,
                                      basis_type=sh_basis,
                                      legacy=is_legacy)
    nib.save(nib.Nifti1Image(priors_3d, img_mask.affine), out_priors)
    del priors_3d

    input_sh_3d = img_sh.get_fdata(dtype=np.float32)
    input_sf_1d = sh_to_sf(input_sh_3d[sub_mask_3d],
                           sphere, sh_order_max=sh_order,
                           basis_type=sh_basis, legacy=is_legacy)

    # Creation of the enhanced-FOD (direction-wise multiplication)
    mult_sf_1d = input_sf_1d * todi_sf
    del todi_sf

    input_max_value = np.max(input_sf_1d, axis=-1, keepdims=True)
    mult_max_value = np.max(mult_sf_1d, axis=-1, keepdims=True)
    mult_positive_mask = np.squeeze(mult_max_value) > 0.0
    mult_sf_1d[mult_positive_mask] = mult_sf_1d[mult_positive_mask] * \
        input_max_value[mult_positive_mask] / \
        mult_max_value[mult_positive_mask]

    # Memory friendly saving
    input_sh_3d[sub_mask_3d] = sf_to_sh(mult_sf_1d, sphere,
                                        sh_order_max=sh_order,
                                        basis_type=sh_basis,
                                        legacy=is_legacy)
    nib.save(nib.Nifti1Image(input_sh_3d, img_mask.affine), out_efod)
    del input_sh_3d

    nib.save(nib.Nifti1Image(sub_mask_3d.astype(
        np.uint8), img_mask.affine), out_todi_mask)

    endpoints_mask = np.zeros(img_mask.shape, dtype=np.uint8)
    sft.to_corner()
    sft.streamlines._data = sft.streamlines._data.astype(np.uint16)
    for streamline in sft.streamlines:
        endpoints_mask[tuple(streamline[0])] = 1
        endpoints_mask[tuple(streamline[-1])] = 1

    nib.save(nib.Nifti1Image(endpoints_mask * mask_data,
                             img_mask.affine), out_endpoints_mask)


if __name__ == "__main__":
    main()
