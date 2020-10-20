#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute neighbors average from fODF
"""

import time
import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_sh_basis_args)

from scilpy.denoise.asym_enhancement import (average_fodf_asymmetrically)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_fodf',
                   help='Path to the input file')

    p.add_argument('out_avafodf',
                   help='Output path of averaged fODF')

    p.add_argument('--nb_iterations', default=1, type=int,
                   help='Number of iterations.')

    p.add_argument('--out_mask',
                   help='Name of output mask. If not specified, the mask is '
                        'saved under wm_mask.nii.gz.')

    p.add_argument('--wm_epsilon', default=1e-16,
                   help='Threshold on WM fODF for output mask.')

    p.add_argument(
        '--sh_order', default=8, type=int,
        help='SH order of the input [%(default)s]')

    p.add_argument(
        '--sphere', default='repulsion724', type=str,
        help='Sphere used for the SH reprojection [%(default)s]'
    )

    p.add_argument(
        '--sharpness', default=1.0, type=float,
        help='Specify sharpness factor to use for weighted average'
        ' [%(default)s]'
    )

    p.add_argument(
        '--sigma', default=1.0, type=float,
        help='Sigma of the gaussian to use [%(default)s]'
    )

    add_sh_basis_args(p)
    add_overwrite_arg(p)

    return p


def get_file_prefix_and_extension(avafodf_file):
    extension_index = avafodf_file.find('.nii')
    if extension_index != -1:
        extension = avafodf_file[extension_index:]
        prefix = avafodf_file[:extension_index]
    else:
        extension = '.nii.gz'
        prefix = avafodf_file
    return prefix, extension


def filter_iterative(fodf, affine, fnames, args):
    for i in range(args.nb_iterations):
        avafodf = average_fodf_asymmetrically(fodf,
                                              sh_order=args.sh_order,
                                              sh_basis=args.sh_basis,
                                              sphere_str=args.sphere,
                                              dot_sharpness=args.sharpness,
                                              sigma=args.sigma)
        nib.save(nib.Nifti1Image(avafodf.astype(np.float), affine),
                 fnames[i])
        fodf = avafodf


def filter_one_shot(fodf, affine, fname, args):
    avafodf = average_fodf_asymmetrically(fodf,
                                          sh_order=args.sh_order,
                                          sh_basis=args.sh_basis,
                                          sphere_str=args.sphere,
                                          dot_sharpness=args.sharpness,
                                          sigma=args.sigma)

    nib.save(nib.Nifti1Image(avafodf.astype(np.float), affine),
             fname)


def generate_nonzero_fodf_mask(fodf, threshold):
    norm = np.linalg.norm(fodf, axis=-1)
    mask = norm > threshold
    return mask


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    inputs = []
    inputs.append(args.in_fodf)

    f_prefix, f_extension = get_file_prefix_and_extension(args.out_avafodf)
    out_mask = f_prefix + '_nonzero_mask' + f_extension
    if args.out_mask:
        out_mask = args.out_mask
    out_avafodf = []
    if args.nb_iterations > 1:
        for i in range(args.nb_iterations):
            outfile = f_prefix + '_{0}'.format(i) + f_extension
            out_avafodf.append(outfile)
    else:
        out_avafodf.append(args.out_avafodf)
    outputs = [out_mask] + out_avafodf

    # Checking args
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, outputs)

    # Prepare data
    fodf_img = nib.nifti1.load(args.in_fodf)
    fodf_data = fodf_img.get_fdata(dtype=np.float)

    # Generate WM fODF mask by applying threshold on fODF amplitude
    mask = generate_nonzero_fodf_mask(fodf_data, args.wm_epsilon)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), fodf_img.affine),
             out_mask)

    # Computing neighbors asymmetric average of fODFs
    t0 = time.perf_counter()
    logging.info('Computing asymmetric averaged fODF')
    if args.nb_iterations > 1:
        filter_iterative(fodf_data, fodf_img.affine, out_avafodf, args)
    else:
        filter_one_shot(fodf_data, fodf_img.affine, out_avafodf[0], args)
    t1 = time.perf_counter()

    elapsedTime = t1 - t0
    logging.info('Elapsed time (s): {0}'.format(elapsedTime))


if __name__ == "__main__":
    main()
