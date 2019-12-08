#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from operator import itemgetter
import os
import shutil

from dipy.segment.clustering import qbx_and_merge
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty)

DESCRIPTION = """
    Compute clusters using QuickBundlesX.
    We cannot know the number of clusters in advance.
"""


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Tractogram filename.\n'
                   'Format must be readable by Nibabel.')
    p.add_argument('dist_thresh', type=float,
                   help='Last QuickBundlesX threshold in mm. Typically \n'
                   'the value are between 10-20mm')
    p.add_argument('output_clusters_dir',
                   help='Path to the clusters directory')

    p.add_argument('--nb_points', type=int, default='20',
                   help='Streamlines will be resampled to have this '
                        'number of points. [%(default)s]')
    p.add_argument('--output_centroids',
                   help='Output tractogram filename.\n'
                   'Format must be readable by the Nibabel API.')

    add_reference(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, [], optional=args.output_centroids)
    if args.output_clusters_dir:
        assert_output_dirs_exist_and_empty(
            parser, args, args.output_clusters_dir)

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
        new_tractogram = nib.streamlines.Tractogram(clusters.centroids,
                                                    affine_to_rasmm=np.eye(4))
        nib.streamlines.save(new_tractogram, args.output_centroids,
                             header=tractogram.header)


if __name__ == "__main__":
    main()
