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

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.streamlines import get_streamlines_bounding_box


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Outlier removal of streamlines',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('bundle',
                   help='Fiber bundle file to remove outliers from.')
    p.add_argument('filtered_bundle',
                   help='Fiber bundle without outliers.')
    p.add_argument('outliers',
                   help='Removed outliers.')
    p.add_argument('--alpha',
                   type=float, default=0.6,
                   help='Percent of the length of the tree that clusters '
                   'of individual streamlines will be pruned.')

    add_overwrite_arg(p)

    return p


def prune(streamlines, threshold, features):
    indices = np.arange(len(streamlines))

    outlier_indices = indices[features < threshold]
    rest_indices = indices[features >= threshold]

    return outlier_indices, rest_indices


def outliers_removal_using_hierarchical_quickbundles(streamlines,
                                                     min_threshold=0.5,
                                                     nb_samplings_max=30,
                                                     sampling_seed=1234):
    if nb_samplings_max < 2:
        raise ValueError("'nb_samplings_max' must be >= 2")

    rng = np.random.RandomState(sampling_seed)
    metric = "MDF_12points"

    box_min, box_max = get_streamlines_bounding_box(streamlines)
    initial_threshold = np.min(np.abs(box_max - box_min)) / 2.

    # Quickbundle's threshold is divided by 1.2 between hierarchical level.
    thresholds = list(takewhile(
        lambda t: t >= min_threshold,
        (initial_threshold / 1.2**i for i in count())))

    ordering = np.arange(len(streamlines))
    path_lengths_per_streamline = 0

    streamlines_path = np.ones((len(streamlines), len(thresholds),
                                nb_samplings_max), dtype=int) * -1

    for i in range(nb_samplings_max):
        rng.shuffle(ordering)

        cluster_orderings = [ordering]
        for j, threshold in enumerate(thresholds):
            id_cluster = 0

            next_cluster_orderings = []
            qb = QuickBundles(metric=metric, threshold=threshold)
            for cluster_ordering in cluster_orderings:
                clusters = qb.cluster(streamlines, ordering=cluster_ordering)

                for k, cluster in enumerate(clusters):
                    streamlines_path[cluster.indices, j, i] = id_cluster
                    id_cluster += 1
                    if len(cluster) > 10:
                        next_cluster_orderings.append(cluster.indices)

            cluster_orderings = next_cluster_orderings

        if i <= 1:  # Needs at least two orderings to compute stderror.
            continue

    path_lengths_per_streamline = np.sum((streamlines_path != -1),
                                         axis=1)[:, :i]

    summary = np.mean(path_lengths_per_streamline,
                      axis=1) / np.max(path_lengths_per_streamline)
    return summary


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.bundle)
    assert_outputs_exist(parser, args, [args.filtered_bundle, args.outliers])
    if args.alpha <= 0 or args.alpha > 1:
        parser.error('--alpha should be ]0, 1]')

    tractogram = nib.streamlines.load(args.bundle)

    if int(tractogram.header['nb_streamlines']) == 0:
        logging.warning("Bundle file contains no streamline")
        return

    streamlines = tractogram.streamlines

    summary = outliers_removal_using_hierarchical_quickbundles(streamlines)
    outliers, outliers_removed = prune(streamlines, args.alpha, summary)

    outliers_cluster = Cluster(indices=outliers, refdata=streamlines)
    outliers_removed_cluster = Cluster(indices=outliers_removed,
                                       refdata=streamlines)

    outliers_data_per_streamline = LazyDict()
    outliers_removed_data_per_streamline = LazyDict()
    for key in tractogram.tractogram.data_per_streamline.keys():
        outliers_data_per_streamline[key] = lambda: [
            tractogram.tractogram.data_per_streamline[key][int(i)]
            for i in outliers]
        outliers_removed_data_per_streamline[key] = lambda: [
            tractogram.tractogram.data_per_streamline[key][int(i)]
            for i in outliers_removed]

    outliers_data_per_point = LazyDict()
    outliers_removed_data_per_point = LazyDict()
    for key in tractogram.tractogram.data_per_point.keys():
        outliers_data_per_point[key] = lambda: [
            tractogram.tractogram.data_per_point[key][int(i)]
            for i in outliers]
        outliers_removed_data_per_point[key] = lambda: [
            tractogram.tractogram.data_per_point[key][int(i)]
            for i in outliers_removed]

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
            args.filtered_bundle,
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
