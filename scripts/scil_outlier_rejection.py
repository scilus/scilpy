#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
from distutils.version import LooseVersion
from itertools import takewhile, count
import logging

from dipy.segment.clustering import QuickBundles, Cluster
import nibabel as nib

from nibabel.streamlines.tractogram import LazyDict, LazyTractogram
import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)
from scilpy.tractanalysis.features import (get_streamlines_bounding_box,
                                           outliers_removal_using_hierarchical_quickbundles,
                                           prune)

DESCRIPTION = """
    Clean a bundle (inliers/outliers) using hiearchical clustering.
"""


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_bundle',
                        help='Fiber bundle file to remove outliers from')
    parser.add_argument('inliers',
                        help='Fiber bundle without outliers')
    parser.add_argument('outliers', help='Removed outliers')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Percent of the length of the tree that clusters '
                        'of individual streamlines will be pruned')
    add_overwrite_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.input_bundle])
    assert_outputs_exists(parser, args, [args.inliers, args.outliers])
    if args.alpha <= 0 or args.alpha > 1:
        parser.error('--alpha should be ]0, 1]')

    tractogram = nib.streamlines.load(args.input_bundle)

    if int(tractogram.header['nb_streamlines']) == 0:
        logging.warning("Bundle file contains no streamline")
        return

    streamlines = tractogram.streamlines

    summary = outliers_removal_using_hierarchical_quickbundles(streamlines)
    outliers, outliers_removed = prune(streamlines, args.alpha, summary)

    outliers_cluster = Cluster(indices=outliers, refdata=streamlines)
    outliers_removed_cluster = Cluster(indices=outliers_removed,
                                       refdata=streamlines)

    outliers_data_per_streamline = tractogram.tractogram.data_per_streamline[outliers]
    
    # outliers_data_per_streamline = LazyDict()
    # outliers_removed_data_per_streamline = LazyDict()
    # for key in tractogram.tractogram.data_per_streamline.keys():
    #     outliers_data_per_streamline[key] = lambda: [
    #         tractogram.tractogram.data_per_streamline[key][int(i)]
    #         for i in outliers]
    #     outliers_removed_data_per_streamline[key] = lambda: [
    #         tractogram.tractogram.data_per_streamline[key][int(i)]
    #         for i in outliers_removed]

    # outliers_data_per_point = LazyDict()
    # outliers_removed_data_per_point = LazyDict()
    # for key in tractogram.tractogram.data_per_point.keys():
    #     outliers_data_per_point[key] = lambda: [
    #         tractogram.tractogram.data_per_point[key][int(i)]
    #         for i in outliers]
    #     outliers_removed_data_per_point[key] = lambda: [
    #         tractogram.tractogram.data_per_point[key][int(i)]
    #         for i in outliers_removed]

    if len(outliers_removed_cluster) == 0:
        print("All streamlines are considered outliers. Please lower the "
              "--alpha parameter")
    else:
        outlier_removed_tractogram = LazyTractogram(
            lambda: outliers_removed_cluster,
            affine_to_rasmm=np.eye(4),
            data_per_streamline=outliers_removed_data_per_streamline,
            data_per_point=outliers_removed_data_per_point)
        nib.streamlines.save(
            outlier_removed_tractogram,
            args.inliers,
            header=tractogram.header)

    if len(outliers_cluster) == 0:
        print("No outlier found. Please raise the --alpha parameter")
    else:
        outlier_tractogram = LazyTractogram(
            lambda: outliers_cluster,
            affine_to_rasmm=np.eye(4),
            data_per_streamline=outliers_data_per_streamline,
            data_per_point=outliers_data_per_point)
        nib.streamlines.save(
            outlier_tractogram,
            args.outliers,
            header=tractogram.header)


if __name__ == '__main__':
    main()
