#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute FA/MD/RD/AD for each tensor solution of MRDS.
It will output the results in 4 different 4D files.
"""

import logging
import numpy as np
import nibabel as nib
import argparse

from dipy.reconst.dti import fractional_anisotropy

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             assert_headers_compatible)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_eigenvalues',
                   help='MRDS eigenvalues file.')

    p.add_argument('--mask',
                   help='Path to a binary mask.\nOnly data inside '
                        'the mask will be used for computations and '
                        'reconstruction. (Default: %(default)s)')

    p.add_argument(
        '--not_all', action='store_true', dest='not_all',
        help='If set, will only save the metrics explicitly specified using '
             'the other metrics flags. (Default: not set).')

    g = p.add_argument_group(title='MRDS-Metrics files flags')
    g.add_argument('--fa', dest='fa', metavar='file', default='',
                   help='Output filename for the FA.')
    g.add_argument('--ad', dest='ad', metavar='file', default='',
                   help='Output filename for the AD.')
    g.add_argument('--rd', dest='rd', metavar='file', default='',
                   help='Output filename for the RD.')
    g.add_argument('--md', dest='md', metavar='file', default='',
                   help='Output filename for the MD.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    if not args.not_all:
        args.fa = args.fa or 'mrds_fa.nii.gz'
        args.ad = args.ad or 'mrds_ad.nii.gz'
        args.rd = args.rd or 'mrds_rd.nii.gz'
        args.md = args.md or 'mrds_md.nii.gz'

    assert_inputs_exist(parser, args.in_eigenvalues, args.mask)
    assert_headers_compatible(parser, args.in_eigenvalues, args.mask)
    assert_outputs_exist(parser, args, [],
                         optional=[args.fa, args.ad, args.rd, args.md])

    eigenvalues_img = nib.load(args.in_eigenvalues)
    lambdas = eigenvalues_img.get_fdata(dtype=np.float32)

    header = eigenvalues_img.header
    affine = eigenvalues_img.affine

    X, Y, Z = lambdas.shape[0:3]

    # load mask
    if args.mask:
        mask = get_data_as_mask(nib.load(args.mask))
    else:
        mask = np.ones((X, Y, Z), dtype=np.uint8)

    fa = np.zeros((X, Y, Z, 3))
    ad = np.zeros((X, Y, Z, 3))
    rd = np.zeros((X, Y, Z, 3))
    md = np.zeros((X, Y, Z, 3))

    if args.fa:
        fa = np.stack((fractional_anisotropy(lambdas[:, :, :, 0:3]),
                       fractional_anisotropy(lambdas[:, :, :, 3:6]),
                       fractional_anisotropy(lambdas[:, :, :, 6:9])),
                      axis=3)
        nib.save(nib.Nifti1Image(np.where(mask[..., None], fa, 0),
                                 affine=affine,
                                 header=header,
                                 dtype=np.float32),
                 args.fa)

    if args.ad:
        ad = np.stack((lambdas[:, :, :, 0],
                       lambdas[:, :, :, 3],
                       lambdas[:, :, :, 6]),
                      axis=3)
        nib.save(nib.Nifti1Image(np.where(mask[..., None], ad, 0),
                                 affine=affine,
                                 header=header,
                                 dtype=np.float32),
                 args.ad)

    if args.rd:
        rd = np.stack(((lambdas[:, :, :, 1] + lambdas[:, :, :, 2])/2,
                       (lambdas[:, :, :, 4] + lambdas[:, :, :, 5])/2,
                       (lambdas[:, :, :, 7] + lambdas[:, :, :, 8])/2),
                      axis=3)
        nib.save(nib.Nifti1Image(np.where(mask[..., None], rd, 0),
                                 affine=affine,
                                 header=header,
                                 dtype=np.float32),
                 args.rd)

    if args.md:
        md = np.stack((np.average(lambdas[:, :, :, 0:3], axis=3),
                       np.average(lambdas[:, :, :, 3:6], axis=3),
                       np.average(lambdas[:, :, :, 6:9], axis=3)),
                      axis=3)
        nib.save(nib.Nifti1Image(np.where(mask[..., None], md, 0),
                                 affine=affine,
                                 header=header,
                                 dtype=np.float32),
                 args.md)


if __name__ == '__main__':
    main()
