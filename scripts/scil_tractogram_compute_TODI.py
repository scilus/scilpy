#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute a Track Orientation Density Image (TODI).

Each segment of the streamlines is weighted by its length (to support
compressed streamlines).

This script can afterwards output a Track Density Image (TDI) or a TODI with SF
or SH representation, based on streamlines' segments.

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

    g = p.add_argument_group("Computing options")
    g.add_argument('--sphere', default='repulsion724',
                   help='Sphere used for the angular discretization. '
                        '[%(default)s]')
    g.add_argument('--mask',
                   help='If set, use the given mask.')
    g.add_argument('--sh_order', type=int, default=8,
                   help='Order of the original SH. [%(default)s]')
    g.add_argument('--normalize_per_voxel', action='store_true',
                   help='If set, normalize each SF/SH at each voxel.')
    gg = g.add_mutually_exclusive_group()
    gg.add_argument('--smooth_todi', action='store_true',
                    help='If set, smooth TODI (angular and spatial).')
    gg.add_argument('--asymmetric', action='store_true',
                    help='If set, compute asymmetric TODI.\n'
                         'Cannot be used with --smooth_todi.')
    g.add_argument('--n_steps', default=1, type=int,
                   help='Number of steps for streamline segments '
                        'subdivision prior to binning [%(default)s].')

    g = p.add_argument_group("Output files. Saves only when filename is set")
    g.add_argument('--out_mask',
                   help='Mask showing where TDI > 0.')
    g.add_argument('--out_tdi',
                   help='Output Track Density Image (TDI).')
    g.add_argument('--out_todi_sf',
                   help='Output TODI, with SF (each directions\n'
                        'on the sphere, requires a lot of memory)')
    g.add_argument('--out_todi_sh',
                   help='Output TODI, with SH coefficients.')

    add_reference_arg(p)
    add_sh_basis_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, args.in_tractogram,
                        [args.mask, args.reference])
    assert_headers_compatible(parser, args.in_tractogram, args.mask,
                              reference=args.reference)
    outputs = [args.out_mask, args.out_tdi, args.out_todi_sf, args.out_todi_sh]
    assert_outputs_exist(parser, args, [], outputs)

    if np.all([f is None for f in outputs]):
        parser.error('No output selected. Choose at least one output option.')

    if args.normalize_per_voxel and not (args.out_todi_sh or args.out_todi_sf):
        logging.warning("Option --normalize_per_voxel is only useful when "
                        "saving output --out_todi_sh or --out_todi_sf. "
                        "Ignoring.")

    # Loading
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    affine, data_shape, _, _ = sft.space_attributes

    sft.to_vox()
    # Because compute_todi expects streamline points (in voxel coordinates)
    # to be in the range [0, size] rather than [-0.5, size - 0.5], we shift
    # the voxel origin to corner.
    sft.to_corner()

    # Processing
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

    if args.normalize_per_voxel:
        # Normalization is done on the SH, but, indirectly, it may change the
        # SF values. So, applying even if we don't have --out_todi_sh.
        todi_obj.normalize_todi_per_voxel()

    # Saving
    logging.info('Saving Outputs ...')
    if args.out_mask:
        img = todi_obj.reshape_to_3d(todi_obj.get_mask())
        img = nib.Nifti1Image(img.astype(np.int16), affine)
        img.to_filename(args.out_mask)

    if args.out_todi_sh:
        sh_basis, is_legacy = parse_sh_basis_arg(args)
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
