#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute the mean/median maximum fODF in the ventricles. The
ventricules are estimated from an MD and FA threshold.

This allows to clip the noise of fODF using an absolute thresold.

--------------------------------------------------------------------------
Reference:
[1] Dell'Acqua, Flavio, et al. "Can spherical deconvolution provide more
    information than fiber orientations? Hindrance modulated orientational
    anisotropy, a true-tract specific index to characterize white matter
    diffusion." Human brain mapping 34.10 (2013): 2464-2483.
--------------------------------------------------------------------------
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg,
                             add_sh_basis_args, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             parse_sh_basis_arg)
from scilpy.reconst.fodf import get_ventricles_max_fodf
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_fodfs',  metavar='fODFs',
                   help='Path of the fODF volume in spherical harmonics (SH).')
    p.add_argument('in_fa',  metavar='FA',
                   help='Path to the FA volume.')
    p.add_argument('in_md',  metavar='MD',
                   help='Path to the mean diffusivity (MD) volume.')

    p.add_argument('--fa_threshold', type=float, default='0.1',
                   help='Maximal threshold of FA (voxels under that threshold '
                        'are considered \nfor evaluation. [%(default)s]).')
    p.add_argument('--md_threshold', type=float, default='0.003',
                   help='Minimal threshold of MD in mm2/s (voxels above that '
                        'threshold are \nconsidered for '
                        'evaluation. [%(default)s]).')
    p.add_argument('--in_mask', metavar='file',
                   help='Path to a binary mask. Only the data inside the '
                        'mask will be used \nfor evaluation. Useful if the '
                        'FA and MD thresholds are not good enough.')
    p.add_argument('--max_value_output',  metavar='file',
                   help='Output path for the text file containing the value. '
                        'If not set the \nfile will not be saved.')
    p.add_argument('--out_mask',  metavar='file',
                   help='Output path for the ventricule mask. If not set, '
                        'the mask \nwill not be saved.')
    p.add_argument('--small_dims',  action='store_true',
                   help='If set, takes the full range of data to search the '
                        'max fodf amplitude \nin ventricles. Useful when the '
                        'data has small dimensions.')
    p.add_argument('--use_median',  action='store_true',
                   help='If set, use the median value instead of the '
                        'mean.')

    add_sh_basis_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_fodfs, args.in_fa, args.in_md],
                        optional=args.in_mask)
    assert_outputs_exist(parser, args, [],
                         [args.max_value_output, args.out_mask])

    # Load input image
    img_fODFs = nib.load(args.in_fodfs)
    fodf = img_fODFs.get_fdata(dtype=np.float32)
    zoom = img_fODFs.header.get_zooms()[:3]

    img_fa = nib.load(args.in_fa)
    fa = img_fa.get_fdata(dtype=np.float32)

    img_md = nib.load(args.in_md)
    md = img_md.get_fdata(dtype=np.float32)

    mask = get_data_as_mask(nib.load(args.in_mask),
                            dtype=bool) if args.in_mask else None

    sh_basis, is_legacy = parse_sh_basis_arg(args)

    value, out_mask = get_ventricles_max_fodf(fodf, fa, md, zoom, sh_basis,
                                              args.fa_threshold,
                                              args.md_threshold,
                                              mask=mask,
                                              small_dims=args.small_dims,
                                              is_legacy=is_legacy,
                                              use_median=args.use_median)

    if args.out_mask:
        img = nib.Nifti1Image(np.array(out_mask, 'float32'),  img_fODFs.affine)
        nib.save(img, args.out_mask)

    if args.max_value_output:
        text_file = open(args.max_value_output, "w")
        text_file.write(str(value))
        text_file.close()
    else:
        print("Maximal value in ventricles: {}".format(value))


if __name__ == "__main__":
    main()
