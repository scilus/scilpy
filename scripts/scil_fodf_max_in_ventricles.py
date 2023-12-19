#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute the maximum fODF in the ventricles. The ventricules are
estimated from a MD and FA threshold.

This allows to clip the noise of fODF using an absolute thresold.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_sh_basis_args, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.reconst.fodf import get_ventricles_max_fodf

EPILOG = """
[1] Dell'Acqua, Flavio, et al. "Can spherical deconvolution provide more
    information than fiber orientations? Hindrance modulated orientational
    anisotropy, a true-tract specific index to characterize white matter
    diffusion." Human brain mapping 34.10 (2013): 2464-2483.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__, epilog=EPILOG)

    p.add_argument('in_fodfs',  metavar='fODFs',
                   help='Path of the fODF volume in spherical harmonics (SH).')
    p.add_argument('in_fa',  metavar='FA',
                   help='Path to the FA volume.')
    p.add_argument('in_md',  metavar='MD',
                   help='Path to the mean diffusivity (MD) volume.')

    p.add_argument('--fa_t', dest='fa_threshold',
                   type=float, default='0.1',
                   help='Maximal threshold of FA (voxels under that threshold'
                        ' are considered for evaluation, [%(default)s]).')
    p.add_argument('--md_t', dest='md_threshold',
                   type=float, default='0.003',
                   help='Minimal threshold of MD in mm2/s (voxels above that '
                        'threshold are considered for '
                        'evaluation, [%(default)s]).')
    p.add_argument('--max_value_output',  metavar='file',
                   help='Output path for the text file containing the value. '
                        'If not set the file will not be saved.')
    p.add_argument('--mask_output',  metavar='file',
                   help='Output path for the ventricule mask. If not set, '
                        'the mask will not be saved.')
    p.add_argument('--small_dims',  action='store_true',
                   help='If set, takes the full range of data to search the '
                        'max fodf amplitude in ventricles. Useful when the '
                        'data is 2D or has small dimensions.')

    add_sh_basis_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_fodfs, args.in_fa, args.in_md])
    assert_outputs_exist(parser, args, [],
                         [args.max_value_output, args.mask_output])

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load input image
    img_fODFs = nib.load(args.in_fodfs)
    fodf = img_fODFs.get_fdata(dtype=np.float32)
    affine = img_fODFs.affine
    zoom = img_fODFs.header.get_zooms()[:3]

    img_fa = nib.load(args.in_fa)
    fa = img_fa.get_fdata(dtype=np.float32)

    img_md = nib.load(args.in_md)
    md = img_md.get_fdata(dtype=np.float32)

    value, mask = get_ventricles_max_fodf(fodf, fa, md, zoom, args)

    if args.mask_output:
        img = nib.Nifti1Image(np.array(mask, 'float32'),  affine)
        nib.save(img, args.mask_output)

    if args.max_value_output:
        text_file = open(args.max_value_output, "w")
        text_file.write(str(value))
        text_file.close()
    else:
        print("Maximal value in ventricles: {}".format(value))


if __name__ == "__main__":
    main()
