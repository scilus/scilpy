#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Compute clusters using QuickBundlesX and save them separately.
    We cannot know the number of clusters in advance.
"""

import argparse
from operator import itemgetter
import os

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.segment.clustering import qbx_and_merge

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Tractogram filename.\n'
                        'Path of the input tractogram or bundle.')
    p.add_argument('dist_thresh', type=float,
                   help='Last QuickBundlesX threshold in mm. Typically \n'
                        'the value are between 10-20mm.')
    p.add_argument('output_clusters_dir',
                   help='Path to the clusters directory.')

    p.add_argument('--nb_points', type=int, default='20',
                   help='Streamlines will be resampled to have this '
                        'number of points [%(default)s].')
    p.add_argument('--output_centroids',
                   help='Output tractogram filename.\n'
                        'Format must be readable by the Nibabel API.')

    add_reference_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, [], optional=args.output_centroids)
    if args.output_clusters_dir:
        assert_output_dirs_exist_and_empty(parser, args,
                                           args.output_clusters_dir,
                                           create_dir=True)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    streamlines = sft.streamlines
    thresholds = [40, 30, 20, args.dist_thresh]
    clusters = qbx_and_merge(streamlines, thresholds,
                             nb_pts=args.nb_points, verbose=False)

    for i, cluster in enumerate(clusters):
        if len(cluster.indices) > 1:
            cluster_streamlines = itemgetter(*cluster.indices)(streamlines)
        else:
            cluster_streamlines = streamlines[cluster.indices]

        new_sft = StatefulTractogram(cluster_streamlines, sft, Space.RASMM)
        save_tractogram(new_sft, os.path.join(args.output_clusters_dir,
                                              'cluster_{}.trk'.format(i)))

    if args.output_centroids:
        new_sft = StatefulTractogram(clusters.centroids, sft, Space.RASMM)
        save_tractogram(new_sft, args.output_centroids)


if __name__ == "__main__":
    main()
