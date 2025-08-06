#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute FA/MD/RD/AD for
each Multi-ResolutionDiscrete-Search (MRDS) solution.
It will output the results in 4 different 4D files.
Each 4th dimension will correspond to each tensor in the MRDS solution.
e.g. FA of tensor D_1 will be in index 0 of the 4th dimension,
     FA of tensor D_2 will be in index 1 of the 4th dimension,
     FA of tensor D_3 will be in index 2 of the 4th dimension.
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
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)
    p.add_argument('in_evals',
                   help='MRDS eigenvalues file (Shape: [X, Y, Z, 9]).\n'
                        'The last dimensions, values 1-3 are associated with '
                        'the first tensor (D_1), 4-6 with the second tensor '
                        '(D_2), 7-9 with the third tensor (D_3).\n'
                        'This file is one of the outputs of '
                        'scil_mrds_select_number_of_tensors.py '
                        '(*_MRDS_evals.nii.gz).')

    p.add_argument('--mask',
                   help='Path to a binary mask.\nOnly data inside '
                        'the mask will be used for computations and '
                        'reconstruction. (Default: %(default)s)')

    p.add_argument(
        '--not_all', action='store_true',
        help='If set, will only save the metrics explicitly specified using '
             'the other metrics flags. (Default: not set).')

    g = p.add_argument_group(title='MRDS-Metrics files flags')
    g.add_argument('--fa', metavar='file', default='',
                   help='Output filename for the FA.')
    g.add_argument('--ad', metavar='file', default='',
                   help='Output filename for the AD.')
    g.add_argument('--rd', metavar='file', default='',
                   help='Output filename for the RD.')
    g.add_argument('--md', metavar='file', default='',
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

    assert_inputs_exist(parser, args.in_evals, args.mask)
    assert_headers_compatible(parser, args.in_evals, args.mask)
    assert_outputs_exist(parser, args, [],
                         optional=[args.fa, args.ad, args.rd, args.md])

    evals_img = nib.load(args.in_evals)
    lambdas = evals_img.get_fdata(dtype=np.float32)

    header = evals_img.header
    affine = evals_img.affine

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
        nib.save(nib.Nifti1Image(fa * mask[..., None],
                                 affine=affine,
                                 header=header,
                                 dtype=np.float32),
                 args.fa)

    if args.ad:
        ad = np.stack((lambdas[:, :, :, 0],
                       lambdas[:, :, :, 3],
                       lambdas[:, :, :, 6]),
                      axis=3)
        nib.save(nib.Nifti1Image(ad * mask[..., None],
                                 affine=affine,
                                 header=header,
                                 dtype=np.float32),
                 args.ad)

    if args.rd:
        rd = np.stack(((lambdas[:, :, :, 1] + lambdas[:, :, :, 2])/2,
                       (lambdas[:, :, :, 4] + lambdas[:, :, :, 5])/2,
                       (lambdas[:, :, :, 7] + lambdas[:, :, :, 8])/2),
                      axis=3)
        nib.save(nib.Nifti1Image(rd * mask[..., None],
                                 affine=affine,
                                 header=header,
                                 dtype=np.float32),
                 args.rd)

    if args.md:
        md = np.stack((np.average(lambdas[:, :, :, 0:3], axis=3),
                       np.average(lambdas[:, :, :, 3:6], axis=3),
                       np.average(lambdas[:, :, :, 6:9], axis=3)),
                      axis=3)
        nib.save(nib.Nifti1Image(md * mask[..., None],
                                 affine=affine,
                                 header=header,
                                 dtype=np.float32),
                 args.md)


if __name__ == '__main__':
    main()
