#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract B0s from DWI, based on the bval and bvec information.

The default behavior is to save the first b0 of the series.

Formerly: scil_extract_b0.py
"""

import argparse
import logging
import os

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs

import nibabel as nib
import numpy as np

from scilpy.dwi.utils import extract_b0
from scilpy.io.utils import (add_b0_thresh_arg, add_overwrite_arg,
                             add_skip_b0_check_arg, add_verbose_arg,
                             assert_inputs_exist)
from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              B0ExtractionStrategy)
from scilpy.utils.filenames import split_name_with_nii

logger = logging.getLogger(__file__)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='DWI Nifti image.')
    p.add_argument('in_bval',
                   help='b-values filename, in FSL format (.bval).')
    p.add_argument('in_bvec',
                   help='b-values filename, in FSL format (.bvec).')
    p.add_argument('out_b0',
                   help='Output b0 file(s).')

    group_ = p.add_argument_group("Options in the case of multiple b0s.")
    group = group_.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true',
                       help='Extract all b0s. Index number will be appended '
                            'to the output file.')
    group.add_argument('--mean', action='store_true', help='Extract mean b0.')
    group.add_argument('--cluster-mean', action='store_true',
                       help='Extract mean of each continuous cluster of b0s.')
    group.add_argument('--cluster-first', action='store_true',
                       help='Extract first b0 of each continuous cluster of '
                            'b0s.')

    p.add_argument('--block-size', '-s',
                   metavar='INT', type=int,
                   help='Load the data using this block size. Useful\nwhen '
                        'the data is too large to be loaded in memory.')

    p.add_argument('--single-image', action='store_true',
                   help='If output b0 volume has multiple time points, only '
                        'outputs a single \nimage instead of a numbered '
                        'series of images.')

    add_b0_thresh_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=True)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _split_time_steps(b0, affine, header, output):
    fname, fext = split_name_with_nii(os.path.basename(output))

    multiple_b0 = b0.shape[-1] > 1
    for t in range(b0.shape[-1]):
        out_name = os.path.join(
            os.path.dirname(os.path.abspath(output)),
            '{}_{}{}'.format(fname, t, fext)) if multiple_b0 else output
        nib.save(nib.Nifti1Image(b0[..., t], affine, header), out_name)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])

    # Outputs are not checked, since multiple use cases
    # are possible and hard to check

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    args.b0_threshold = check_b0_threshold(bvals.min(),
                                           b0_thr=args.b0_threshold,
                                           skip_b0_check=args.skip_b0_check)
    gtab = gradient_table(bvals, bvecs, b0_threshold=args.b0_threshold)
    b0_idx = np.where(gtab.b0s_mask)[0]

    logger.info('Number of b0 images in the data: {}'.format(len(b0_idx)))

    strategy, extract_in_cluster = B0ExtractionStrategy.FIRST, False
    if args.mean or args.cluster_mean:
        strategy = B0ExtractionStrategy.MEAN
        extract_in_cluster = args.cluster_mean
    elif args.all:
        strategy = B0ExtractionStrategy.ALL
    elif args.cluster_first:
        extract_in_cluster = True

    image = nib.load(args.in_dwi)

    b0_volumes = extract_b0(
        image, gtab.b0s_mask, extract_in_cluster, strategy, args.block_size)

    if len(b0_volumes.shape) > 3 and not args.single_image:
        _split_time_steps(b0_volumes, image.affine, image.header, args.out_b0)
    else:
        nib.save(nib.Nifti1Image(b0_volumes, image.affine, image.header),
                 args.out_b0)


if __name__ == '__main__':
    main()
