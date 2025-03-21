#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute clusters using QuickBundlesX and save them separately.
We cannot know the number of clusters in advance.

Quickbundles:
Garyfallidis, E. et al. (2012). Quickbundles, a method for tractography
simplification. Frontiers in neuroscience, 6, 175.

QuickbundlesX:
Garyfallidis, E. et al. (2016) QuickBundlesX: sequential clustering of millions
of streamlines in multiple levels of detail at record execution time. 24th
International Society of Magnetic Resonance in Medicine (ISMRM).

"QuickBundlesX shows a remarkable 20+X speedup over its predecessor"

Formerly: scil_compute_qbx.py
"""

import argparse
import logging
from operator import itemgetter
import os

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.segment.clustering import qbx_and_merge

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             add_verbose_arg)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractogram',
                   help='Tractogram filename.\n'
                        'Path of the input tractogram or bundle.')
    p.add_argument('dist_thresh', type=float,
                   help='Last QuickBundlesX threshold in mm. Typically \n'
                        'the value are between 10-20mm.')
    p.add_argument('out_clusters_dir',
                   help='Path where to save the clusters directory.')

    p.add_argument('--nb_points', type=int, default='20',
                   help='Streamlines will be resampled to have this '
                        'number of points [%(default)s].')
    p.add_argument('--out_centroids',
                   help='Output tractogram filename.\n'
                        'Format must be readable by the Nibabel API.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_tractogram, args.reference)
    assert_outputs_exist(parser, args, [], optional=args.out_centroids)
    assert_output_dirs_exist_and_empty(parser, args,
                                       args.out_clusters_dir,
                                       create_dir=True)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    thresholds = [40, 30, 20, args.dist_thresh]
    clusters = qbx_and_merge(sft.streamlines, thresholds,
                             nb_pts=args.nb_points, verbose=False)

    logging.info("Tractogram was separated into {} clusters. Saving..."
                 .format(len(clusters)))

    for i, cluster in enumerate(clusters):
        if len(cluster.indices) > 1:
            cluster_streamlines = itemgetter(*cluster.indices)(sft.streamlines)
        else:
            cluster_streamlines = sft.streamlines[cluster.indices]

        new_sft = StatefulTractogram.from_sft(cluster_streamlines, sft)
        save_tractogram(new_sft, os.path.join(args.out_clusters_dir,
                                              'cluster_{}.trk'.format(i)))

    if args.out_centroids:
        new_sft = StatefulTractogram.from_sft(clusters.centroids, sft)
        save_tractogram(new_sft, args.out_centroids)


if __name__ == "__main__":
    main()
