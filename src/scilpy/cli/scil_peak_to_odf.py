#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Convert principal directions (peaks) to orientation distribution
functions expressed in spherical harmonics coefficients.
"""
import argparse
import logging
import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.core.sphere import hemi_icosahedron
from tqdm import tqdm
from scilpy.reconst.sh import generate_apodized_delta_kernel
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg, add_verbose_arg,
                             add_sh_basis_args, parse_sh_basis_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_peaks', help='Input peaks nifti image.')
    p.add_argument('out_sh',
                   help='Output spherical harmonics (hist-FOD) image.')

    p.add_argument('--brain_mask',
                   help='Optional nifti image to mask the output hist-FOD.\n'
                        'Only non-zero voxels will be evaluated.')
    p.add_argument('--sh_order_max', type=int, default=8,
                   help='SH order for hist-FOD. [%(default)s]')
    add_sh_basis_args(p)
    p.add_argument('--disable_apodization', action='store_true',
                   help='Disable apodized delta kernel for '
                        'mapping peaks to SH coefficients. [%(default)s]')

    p.add_argument('--width', type=int, default=7,
                   help='Width in voxels for average smoothing. Can be set to 1 to disable smoothing. [%(default)s]')
    p.add_argument('--padding_mode', choices=['constant', 'reflect', 'nearest'], default='constant',
                   help='Mode for padding. [%(default)s]')
    p.add_argument('--padding_cval', type=float, default=0.0,
                   help='Value for padding. [%(default)s]')
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_peaks)
    assert_outputs_exist(parser, args, args.out_sh)

    basis_type, legacy = parse_sh_basis_arg(args)
    brain_mask = None
    if args.brain_mask is not None:
        mask_im = nib.load(args.brain_mask)
        brain_mask = mask_im.get_fdata().astype(bool)

    apodize_kernel = None
    if not args.disable_apodization:
        apodize_kernel = generate_apodized_delta_kernel(
            args.sh_order_max, basis_type, legacy)

    sphere = hemi_icosahedron.subdivide(n=4)
    sh_to_sf_mat = sh_to_sf_matrix(sphere, sh_order_max=args.sh_order_max,
                                   basis_type=basis_type, legacy=legacy,
                                   return_inv=False).astype(np.float32)

    peaks_im = nib.load(args.in_peaks)
    peaks = peaks_im.get_fdata().astype(np.float32)  # force float32 to save memory

    # numpy.zeros does not actually allocate memory until values are assigned.
    out_sh = np.zeros(peaks_im.shape[:-1] + (sh_to_sf_mat.shape[0],),
                      dtype=np.float32)
    out_sh[:] = 0  # force allocation of memory

    # process peak
    logging.info("Processing peaks to SH coefficients...")
    peaks_norm = np.linalg.norm(peaks, axis=-1)
    mask = (peaks_norm > 0)
    if brain_mask is not None:
        mask = mask & brain_mask
    peaks1d = peaks[mask]

    peaks1d_to_sph_ind = np.zeros((peaks1d.shape[0],), dtype=int)
    max_dot = np.zeros((peaks1d.shape[0],), dtype=float)
    for vert_idx, d in enumerate(tqdm(sphere.vertices, desc="Processing sphere vertices")):
        dot = np.abs(peaks1d.dot(d))
        update = dot > max_dot
        max_dot[update] = dot[update]
        peaks1d_to_sph_ind[update] = vert_idx

    # THIS LINE IS VERY MEMORY INTENSIVE: OOM ON LARGE IMAGES
    sh = sh_to_sf_mat.T[peaks1d_to_sph_ind]
    if apodize_kernel is not None:
        sh = sh * apodize_kernel

    out_sh[mask] = sh

    if args.width > 1:
        logging.info("Post-filtering of SH coefficients...")
        out_sh = uniform_filter(out_sh, size=args.width, mode=args.padding_mode,
                                cval=args.padding_cval, axes=(0, 1, 2))

    # scale by voxel size for compatibility with MRtrix
    vox_res = np.mean(peaks_im.header.get_zooms()[:3])
    out_sh = out_sh * vox_res

    # Mask output if brain mask provided
    if brain_mask is not None:
        out_sh[~brain_mask] = 0

    logging.info("Saving output...")
    nib.save(nib.Nifti1Image(out_sh.astype(np.float32),
                             peaks_im.affine), args.out_sh)


if __name__ == '__main__':
    main()
