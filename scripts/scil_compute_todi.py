#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference,
                             add_sh_basis_args,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.tractanalysis.todi import TrackOrientationDensityImaging


DESCRIPTION = """
    Compute a length-weighted Track Orientation Density Image (TODI).
    This script can afterwards output a length-weighted Track Density Image
    (TDI) or a length-weighted TODI, based on streamlines' segments.\n\n
    """

EPILOG = """
    References:
        [1] Dhollander T, Emsell L, Van Hecke W, Maes F, Sunaert S, Suetens P.
            Track orientation density imaging (TODI) and
            track orientation distribution (TOD) based tractography.
            NeuroImage. 2014 Jul 1;94:312-36.
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tract_filename',
                   help='Input streamlines file.')

    add_reference(p)

    p.add_argument('--sphere', default='repulsion724',
                   help='sphere used for the angular discretization.')

    p.add_argument('--mask',
                   help='Use the given mask')

    p.add_argument('--out_mask',
                   help='Mask showing where TDI > 0.')

    p.add_argument('--out_lw_tdi',
                   help='Output length-weighted TDI map.')

    p.add_argument('--out_lw_todi',
                   help='Output length-weighted TODI map.')

    p.add_argument('--out_lw_todi_sh',
                   help='Output length-weighted TODI map, '
                   'with SH coefficient.')

    p.add_argument('--sh_order', type=int, default=8,
                   help='Order of the original SH.')

    p.add_argument('--sh_normed', action='store_true',
                   help='Normalize sh.')

    p.add_argument('--smooth', action='store_true',
                   help='Smooth todi (angular and spatial).')

    add_sh_basis_args(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, args.tract_filename,
                        [args.mask, args.reference])

    output_file_list = []
    if args.out_mask:
        output_file_list.append(args.out_mask)
    if args.out_lw_tdi:
        output_file_list.append(args.out_lw_tdi)
    if args.out_lw_todi:
        output_file_list.append(args.out_lw_todi)
    if args.out_lw_todi_sh:
        output_file_list.append(args.out_lw_todi_sh)

    if not output_file_list:
        parser.error('No output to be done')

    assert_outputs_exist(parser, args, output_file_list)

    sft = load_tractogram_with_reference(parser, args, args.tract_filename)
    affine, data_shape, _, _ = sft.space_attribute
    sft.to_vox()

    logging.info('Computing length-weighted TODI ...')
    todi_obj = TrackOrientationDensityImaging(tuple(data_shape), args.sphere)
    todi_obj.compute_todi(sft.streamlines, length_weights=True)

    if args.smooth:
        logging.info('Smoothing ...')
        todi_obj.smooth_todi_dir()
        todi_obj.smooth_todi_spatial()

    if args.mask:
        mask = nib.load(args.mask).get_data()
        todi_obj.mask_todi(mask)

    logging.info('Saving Outputs ...')
    if args.out_mask:
        data = todi_obj.get_mask()
        img = todi_obj.reshape_to_3d(data)
        img = nib.Nifti1Image(img.astype(np.int16), affine)
        img.to_filename(args.out_mask)

    if args.out_lw_todi_sh:
        if args.sh_normed:
            todi_obj.normalize_todi_per_voxel()
        img = todi_obj.get_sh(args.sh_basis, args.sh_order)
        img = todi_obj.reshape_to_3d(img)
        img = nib.Nifti1Image(img.astype(np.float32), affine)
        img.to_filename(args.out_lw_todi_sh)

    if args.out_lw_tdi:
        img = todi_obj.get_tdi()
        img = todi_obj.reshape_to_3d(img)
        img = nib.Nifti1Image(img.astype(np.float32), affine)
        img.to_filename(args.out_lw_tdi)

    if args.out_lw_todi:
        img = todi_obj.get_todi()
        img = todi_obj.reshape_to_3d(img)
        img = nib.Nifti1Image(img.astype(np.float32), affine)
        img.to_filename(args.out_lw_todi)


if __name__ == '__main__':
    main()
