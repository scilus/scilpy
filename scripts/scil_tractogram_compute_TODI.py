#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute a Track Orientation Density Image (TODI).
Each segment of the streamlines is weighted by its length
(to support compressed streamlines).
This script can afterwards output a Track Density Image (TDI)
or a TODI with SF or SH representation, based on streamlines' segments.

Formerly: scil_compute_todi.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_sh_basis_args, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             parse_sh_basis_arg, assert_headers_compatible)
from scilpy.tractanalysis.todi import TrackOrientationDensityImaging


EPILOG = """
    References:
        [1] Dhollander T, Emsell L, Van Hecke W, Maes F, Sunaert S, Suetens P.
            Track orientation density imaging (TODI) and
            track orientation distribution (TOD) based tractography.
            NeuroImage. 2014 Jul 1;94:312-36.
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Input streamlines file.')

    add_reference_arg(p)

    p.add_argument('--sphere', default='repulsion724',
                   help='sphere used for the angular discretization. '
                        '[%(default)s]')

    p.add_argument('--mask',
                   help='Use the given mask.')

    p.add_argument('--out_mask',
                   help='Mask showing where TDI > 0.')

    p.add_argument('--out_tdi',
                   help='Output Track Density Image (TDI).')

    p.add_argument('--out_todi_sf',
                   help='Output TODI, with SF (each directions\n'
                        'on the sphere, requires a lot of memory)')

    p.add_argument('--out_todi_sh',
                   help='Output TODI, with SH coefficients.')

    p.add_argument('--sh_order', type=int, default=8,
                   help='Order of the original SH. [%(default)s]')

    p.add_argument('--normalize_per_voxel', action='store_true',
                   help='Normalize each SF/SH at each voxel [%(default)s].')

    p.add_argument('--smooth_todi', action='store_true',
                   help='Smooth TODI (angular and spatial) [%(default)s].')

    p.add_argument('--asymmetric', action='store_true',
                   help='Compute asymmetric TODI [%(default)s].')

    p.add_argument('--n_steps', default=1, type=int,
                   help='Number of steps for streamline segments '
                        'subdivision prior to binning [%(default)s].')

    add_sh_basis_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram,
                        [args.mask, args.reference])
    assert_headers_compatible(parser, args.in_tractogram, args.mask,
                              reference=args.reference)

    output_file_list = []
    if args.out_mask:
        output_file_list.append(args.out_mask)
    if args.out_tdi:
        output_file_list.append(args.out_tdi)
    if args.out_todi_sf:
        output_file_list.append(args.out_todi_sf)
    if args.out_todi_sh:
        output_file_list.append(args.out_todi_sh)

    if not output_file_list:
        parser.error('No output to be done')

    if args.smooth_todi and args.asymmetric:
        parser.error('Invalid arguments combination. '
                     'Cannot smooth asymmetric TODI.')

    assert_outputs_exist(parser, args, output_file_list)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    affine, data_shape, _, _ = sft.space_attributes

    sft.to_vox()
    # Because compute_todi expects streamline points (in voxel coordinates)
    # to be in the range (0..size) rather than (-0.5..size - 0.5), we shift
    # the voxel origin to corner (will only be done if it's not already the
    # case).
    sft.to_corner()

    logging.info('Computing length-weighted TODI ...')
    todi_obj = TrackOrientationDensityImaging(tuple(data_shape), args.sphere)
    todi_obj.compute_todi(sft.streamlines, length_weights=True,
                          n_steps=args.n_steps, asymmetric=args.asymmetric)

    if args.smooth_todi:
        logging.info('Smoothing ...')
        todi_obj.smooth_todi_dir()
        todi_obj.smooth_todi_spatial()

    if args.mask:
        mask = get_data_as_mask(nib.load(args.mask))
        todi_obj.mask_todi(mask)

    logging.info('Saving Outputs ...')
    if args.out_mask:
        data = todi_obj.get_mask()
        img = todi_obj.reshape_to_3d(data)
        img = nib.Nifti1Image(img.astype(np.int16), affine)
        img.to_filename(args.out_mask)

    if args.out_todi_sh:
        sh_basis, is_legacy = parse_sh_basis_arg(args)
        if args.normalize_per_voxel:
            todi_obj.normalize_todi_per_voxel()
        img = todi_obj.get_sh(sh_basis, args.sh_order,
                              full_basis=args.asymmetric,
                              is_legacy=is_legacy)
        img = todi_obj.reshape_to_3d(img)
        img = nib.Nifti1Image(img.astype(np.float32), affine)
        img.to_filename(args.out_todi_sh)

    if args.out_tdi:
        img = todi_obj.get_tdi()
        img = todi_obj.reshape_to_3d(img)
        img = nib.Nifti1Image(img.astype(np.float32), affine)
        img.to_filename(args.out_tdi)

    if args.out_todi_sf:
        img = todi_obj.get_todi()
        img = todi_obj.reshape_to_3d(img)
        img = nib.Nifti1Image(img.astype(np.float32), affine)
        img.to_filename(args.out_todi_sf)


if __name__ == '__main__':
    main()
