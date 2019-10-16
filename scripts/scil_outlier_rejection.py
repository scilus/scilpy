#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

import nibabel as nib
from nibabel.streamlines.tractogram import Tractogram
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             check_tracts_same_format)
from scilpy.tractanalysis.features import (
    outliers_removal_using_hierarchical_quickbundles,
    prune)

DESCRIPTION = """
Clean a bundle (inliers/outliers) using hiearchical clustering.
http://archive.ismrm.org/2015/2844.html
"""


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('in_bundle',
                        help='Fiber bundle file to remove outliers from.')
    parser.add_argument('out_bundle',
                        help='Fiber bundle without outliers.')
    parser.add_argument('--remaining_bundle',
                        help='Removed outliers.')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Percent of the length of the tree that clusters '
                             'of individual streamlines will be pruned.')
    add_overwrite_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundle)
    assert_outputs_exist(parser, args, args.out_bundle, args.remaining_bundle)
    if args.alpha <= 0 or args.alpha > 1:
        parser.error('--alpha should be ]0, 1]')

    tractogram = nib.streamlines.load(args.in_bundle)

    if int(tractogram.header['nb_streamlines']) == 0:
        logging.warning("Bundle file contains no streamline")
        return

    check_tracts_same_format(parser, [args.in_bundle, args.out_bundle,
                                      args.remaining_bundle])

    streamlines = tractogram.streamlines

    summary = outliers_removal_using_hierarchical_quickbundles(streamlines)
    outliers, inliers = prune(streamlines, args.alpha, summary)

    inliers_streamlines = tractogram.streamlines[inliers]
    inliers_data_per_streamline = tractogram.tractogram.data_per_streamline[inliers]
    inliers_data_per_point = tractogram.tractogram.data_per_point[inliers]

    outliers_streamlines = tractogram.streamlines[outliers]
    outliers_data_per_streamline = tractogram.tractogram.data_per_streamline[outliers]
    outliers_data_per_point = tractogram.tractogram.data_per_point[outliers]

    if len(inliers_streamlines) == 0:
        logging.warning("All streamlines are considered outliers."
                        "Please lower the --alpha parameter")
    else:
        inliers_tractogram = Tractogram(
            inliers_streamlines,
            affine_to_rasmm=np.eye(4),
            data_per_streamline=inliers_data_per_streamline,
            data_per_point=inliers_data_per_point)
        nib.streamlines.save(inliers_tractogram, args.out_bundle,
                             header=tractogram.header)

    if len(outliers_streamlines) == 0:
        logging.warning("No outlier found. Please raise the --alpha parameter")
    elif args.remaining_bundle:
        outlier_tractogram = Tractogram(
            outliers_streamlines,
            affine_to_rasmm=np.eye(4),
            data_per_streamline=outliers_data_per_streamline,
            data_per_point=outliers_data_per_point)
        nib.streamlines.save(outlier_tractogram, args.remaining_bundle,
                             header=tractogram.header)


if __name__ == '__main__':
    main()
