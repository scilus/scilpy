#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Remove very similar streamlines from a bundle.
Uses clustering to speed up the process. Streamlines are considered as similar
based on a MDF threshold within each cluster. Can be used with large bundles,
but the clustering parameters will need to be adjusted.

The algorithm still uses a system of chunks to ensure the amount of comparison
(n**2) does not grow out of control. To overcome limitations related to this
use of chunks, multiple iterations must be done until a convergence threshold
is achieved.

The subsampling threshold should be between 2mm and 5mm, 5mm being quite
aggressive. A CST where all fanning streamlines are important should be around
2mm, while an AF can go around 4mm.

The --processes parameters should only be use on massive bundle. For example,
100 000 streamlines can be split among 8 processes.
"""


import argparse
from itertools import repeat, chain
import logging
import multiprocessing
import random
from time import time

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from nibabel.streamlines import ArraySequence

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.segment.models import subsample_clusters
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from dipy.segment.clustering import qbx_and_merge


def multiprocess_subsampling(args):
    streamlines = args[0]
    min_distance = args[1]
    cluster_thr = args[2]
    min_cluster_size = args[3]
    average_streamlines = args[4]

    min_cluster_size = max(min_cluster_size, 1)
    thresholds = [40, 30, 20, cluster_thr]
    cluster_map = qbx_and_merge(ArraySequence(streamlines),
                                thresholds,
                                nb_pts=20,
                                verbose=False)

    return subsample_clusters(cluster_map, streamlines, min_distance,
                              min_cluster_size, average_streamlines)


def _buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_bundle',
                   help='Path of the input bundle.')

    p.add_argument('min_distance', type=float,
                   help='Distance threshold for 2 streamlines to be '
                        'considered similar (mm).')

    p.add_argument('out_bundle',
                   help='Path of the output tractography file')

    p.add_argument('--clustering_thr', type=float, default=6,
                   help='Clustering threshold for QB/QBx (mm), during '
                        'the first approximation [%(default)s].')
    p.add_argument('--min_cluster_size', type=int, default=5,
                   help='Minimum cluster size for the first iteration '
                        '[%(default)s].')
    p.add_argument('--convergence', type=int, default=100,
                   help='Streamlines count difference threshold to stop '
                        're-running the algorithm [%(default)s].')
    p.add_argument('--avg_similar', action='store_true',
                   help='Average similar streamlines rather than removing them '
                        '[%(default)s]. Requires a small min_distance. '
                        'Allows for some smoothing.')
    p.add_argument('--processes', type=int, default=1,
                   help='Number of desired processes [%(default)s].')

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundle)
    assert_outputs_exist(parser, args, args.out_bundle)

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    streamlines = list(sft.streamlines)
    original_length = len(streamlines)
    logging.debug('Loaded %s streamlines...', original_length)

    pool = multiprocessing.Pool(args.processes)
    timer = time()

    logging.debug('Lauching subsampling on %s processes.', args.processes)
    last_iteration = False
    while True:
        if len(streamlines) < 1000:
            logging.warning('Subsampling less than 1000 streamlines is risky.')
            break
        current_iteration_length = len(streamlines)
        skip = int(len(streamlines) / args.processes) + 1

        # Cheap trick to avoid duplication in memory, the pop removes from
        # one list to append it to the other, slower but allows bigger bundles
        split_streamlines_list = []
        for i in range(args.processes):
            split_streamlines_list.append(streamlines[0:skip])
            del streamlines[0:skip]

        resulting_streamlines = pool.map(multiprocess_subsampling,
                                         zip(split_streamlines_list,
                                             repeat(args.min_distance),
                                             repeat(args.clustering_thr),
                                             repeat(args.min_cluster_size),
                                             repeat(args.avg_similar)))

        # Fused all subprocesses' result together
        streamlines = list(chain(*resulting_streamlines))
        difference_length = current_iteration_length - len(streamlines)
        logging.debug('Difference (before - after): %s streamlines were removed',
                      difference_length)

        if last_iteration and difference_length < args.convergence:
            logging.debug('Before (%s)-> After (%s), total runtime of %s sec.',
                          original_length, len(streamlines),
                          round(time() - timer, 3))
            break
        elif difference_length < args.convergence:
            logging.debug('The smart-subsampling converged, below %s '
                          'different streamlines. Adding single-thread iteration.',
                          args.convergence)
            args.processes = 1
            last_iteration = True
        else:
            logging.debug('Threshold of convergence was not achieved.'
                          ' Need another run...\n')
            args.min_cluster_size = 1

            # Once the streamlines reached a low enough amount, switch to
            # single thread for full comparison
            if len(streamlines) < 10000:
                args.processes = 1
            random.shuffle(streamlines)

    # After convergence, we can simply save the output
    new_sft = StatefulTractogram.from_sft(streamlines, sft)
    save_tractogram(new_sft, args.out_bundle)


if __name__ == "__main__":
    main()
