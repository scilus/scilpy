#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

from dipy.tracking.streamline import transform_streamlines
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             assert_inputs_exist, assert_outputs_exists)
from scilpy.tractanalysis.todi import TrackOrientationDensityImaging
import scilpy.tractanalysis.todi_util as todi_u


DESCRIPTION = """
    Compute a length-weighted Track Orientation Density Image (TODI).
    This script can afterwards output a length-weighted Density image (TDI)
    or a length-weighted TODI, based on streamlines' segments.\n\n
    """

EPILOG = """
    References:
        [1] Dhollander T, Emsell L, Van Hecke W, Maes F, Sunaert S, Suetens P.
            Track orientation density imaging (TODI) and
            track orientation distribution (TOD) based tractography.
            NeuroImage. 2014 Jul 1;94:312-36.
    """


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tract_filename',
                   help="Streamlines file.")

    p.add_argument('--sphere', default='repulsion724',
                   help="sphere used for the angular discretization")

    p.add_argument('--mask',
                   help='Use the given mask')

    p.add_argument('--min_length', default=None, type=float,
                   help='streamlines minimum length in mm')

    p.add_argument('--max_length', default=None, type=float,
                   help='streamlines maximum length in mm')

    p.add_argument('--resample_mm', default=0.2, type=float,
                   help='streamlines resample step size (precision in mm)\n'
                   ' [default: %(default)s]')

    p.add_argument('--out_mask',
                   help='Mask showing where TDI > 0')

    p.add_argument('--out_lw_tdi',
                   help='Output length-weighted TDI map')

    p.add_argument('--out_lw_todi',
                   help='Output length-weighted TODI map')

    p.add_argument('--out_lw_todi_sh',
                   help='Output length-weighted TODI map, with SH coefficient')

    p.add_argument('--sh_order', type=int, default=8,
                   help='Order of the original SH')

    p.add_argument('--sh_normed', action='store_true',
                   help='normalize sh')

    p.add_argument('--smooth', action='store_true',
                   help='smooth todi (angular and spatial)')

    add_sh_basis_args(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, required=[args.tract_filename],
                        optional=[args.mask])
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
    else:
        assert_outputs_exists(parser, args, output_file_list)

    trk_f = nib.streamlines.load(args.tract_filename)
    affine = trk_f.header["voxel_to_rasmm"]
    data_shape = tuple(trk_f.header["dimensions"][:3])
    inv_affine = np.linalg.inv(affine)
    inv_affine[:3, 3] += 0.5

    streamlines = todi_u.filter_streamlines(
        trk_f.streamlines, args.min_length, args.max_length, args.resample_mm)
    streamlines = transform_streamlines(streamlines, inv_affine)

    logging.info('Computing length-weighted TODI ...')
    todi_obj = TrackOrientationDensityImaging(data_shape, args.sphere)
    todi_obj.compute_todi(streamlines, length_weights=True)

    if args.smooth:
        logging.info('Smoothing ...')
        todi_obj.smooth_todi_dir()
        todi_obj.smooth_todi_spatial()

    if args.mask:
        mask = nib.load(args.mask).get_data()
        todi_obj.mask_todi(mask)

    logging.info('Saving Outputs ...')
    if args.out_mask:
        data = todi_obj.get_mask().astype(np.int16)
        img = todi_obj.reshape_to_3d(data)
        img = nib.Nifti1Image(img, affine)
        img.to_filename(args.out_mask)

    if args.out_lw_todi_sh:
        img = todi_obj.get_todi().astype(np.float32)
        img = todi_obj.get_sh(img, args.sh_basis, args.sh_order,
                              args.sh_normed)
        img = todi_obj.reshape_to_3d(img)
        img = nib.Nifti1Image(img, affine)
        img.to_filename(args.out_lw_todi_sh)

    if args.out_lw_tdi:
        img = todi_obj.get_tdi().astype(np.float32)
        img = todi_obj.reshape_to_3d(img)
        img = nib.Nifti1Image(img, affine)
        img.to_filename(args.out_lw_tdi)

    if args.out_lw_todi:
        img = todi_obj.get_todi().astype(np.float32)
        img = todi_obj.reshape_to_3d(img)
        img = nib.Nifti1Image(img, affine)
        img.to_filename(args.out_lw_todi)


if __name__ == "__main__":
    main()
