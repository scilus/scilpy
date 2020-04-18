#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Compute clusters using QuickBundlesX and save them separately.
    We cannot know the number of clusters in advance.
"""

import argparse
import itertools
import multiprocessing

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.segment.clustering import qbx_and_merge
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             add_reference_arg,
                             add_seed_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.segment.voting_scheme import streamlines_to_memmap
from scilpy.segment.recobundlesx import reconstruct_streamlines
from scilpy.tractanalysis.features import (
    outliers_removal_using_hierarchical_quickbundles,
    prune)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram',
                   help='Tractogram filename.\n'
                        'Path of the input tractogram or bundle.')
    p.add_argument('dist_thresh', type=float,
                   help='Last QuickBundlesX threshold in mm. Typically \n'
                        'the value are between 10-20mm.')
    p.add_argument('inliers',
                   help='Path of the output inliers bundle file.')

    p.add_argument('--outliers',
                   help='Path of the output outliers bundle file.')
    p.add_argument('--min_cluster_size', type=int, default=2,
                   help='Minimum cluster size for consideration [%(default)s].\n'
                        'Must be at least 1.')
    p.add_argument('--alpha', type=float, default=0.6,
                   help='Percent of the length of the tree that clusters '
                        'of individual streamlines will be pruned '
                        '[%(default)s].')

    add_processes_arg(p)
    add_seed_arg(p)
    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def outliers_removal_wrapper(args):
    cluster_indices = args[0]
    memmap_filenames = args[1]
    alpha = args[2]
    min_clusters_size = args[3]
    if len(cluster_indices) < min_clusters_size:
        return []

    cluster_streamlines = reconstruct_streamlines(memmap_filenames,
                                                  cluster_indices)

    summary = outliers_removal_using_hierarchical_quickbundles(
        cluster_streamlines)
    _, cluster_inliers = prune(cluster_streamlines, alpha, summary)

    return cluster_indices[cluster_inliers]


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.inliers, args.outliers)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    thresholds = [40, 30, 20, args.dist_thresh]
    rng = np.random.RandomState(args.seed)
    clusters_map = qbx_and_merge(sft.streamlines, thresholds,
                                 nb_pts=2, rng=rng,
                                 verbose=False)

    indices = [np.asarray(cluster.indices, dtype=np.int32)
               for cluster in clusters_map]
    _, tmp_filenames = streamlines_to_memmap(sft.streamlines)

    pool = multiprocessing.Pool(args.nbr_processes)
    results = pool.map(outliers_removal_wrapper,
                       zip(indices,
                           itertools.repeat(tmp_filenames),
                           itertools.repeat(args.alpha),
                           itertools.repeat(args.min_cluster_size)))
    pool.close()
    pool.join()

    inliers = np.fromiter(itertools.chain(*results), dtype=np.int32)
    outliers = np.setdiff1d(range(len(sft)), inliers)

    inliers_streamlines = sft.streamlines[inliers]
    inliers_dps = sft.data_per_streamline[inliers]
    inliers_dpp = sft.data_per_point[inliers]
    new_sft = StatefulTractogram.from_sft(inliers_streamlines, sft,
                                          data_per_streamline=inliers_dps,
                                          data_per_point=inliers_dpp)
    save_tractogram(new_sft, args.inliers)
    if args.outliers:
        outliers_streamlines = sft.streamlines[outliers]
        outliers_dps = sft.data_per_streamline[outliers]
        outliers_dpp = sft.data_per_point[outliers]
        new_sft = StatefulTractogram.from_sft(outliers_streamlines, sft,
                                              data_per_streamline=outliers_dps,
                                              data_per_point=outliers_dpp)
        save_tractogram(new_sft, args.outliers)


if __name__ == "__main__":
    main()
