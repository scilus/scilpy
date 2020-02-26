#!/usr/bin/env python

import argparse
from itertools import repeat, chain
import multiprocessing
import os
import random
from time import time

import numpy as np
import nibabel as nib

from scilpy.segment.models import subsample_clusters
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
assert_inputs_exist,
                             assert_outputs_exist)
from dipy.segment.clustering import qbx_and_merge

DESCRIPTION = """
Using clusters to speed it up, remove very similar streamlines within the
same clusters using a MDF threshold. Can be used with big bundle thanks to
QBx, but if clusters (even some of them have 5000+ streamlines the time
needed to create a n**2 will grow out of control ...
"""

def multiprocess_subsampling(args):
    streamlines = args[0]
    min_distance = args[1]
    cluster_thr = args[2]
    min_cluster_size = args[3]
    average_streamlines = args[4]

    min_cluster_size = max(min_cluster_size, 1)
    thresholds = [40, 30, 20, cluster_thr]
    cluster_map = qbx_and_merge(streamlines, thresholds, verbose=False)

    return subsample_clusters(cluster_map, streamlines, min_distance,
                              min_cluster_size, average_streamlines)


def _buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)

    p.add_argument('in_bundle',
                   help='Path of the input bundle.')

    p.add_argument('min_distance', type=float, 
                   help='Distance threshold for 2 streamlines to be '
                        'considered similar (mm)')

    p.add_argument('out_bundle', 
                   help='Path of the output tractography file')

    p.add_argument('--clustering_thr', type=float, default=6,
                   help='Clustering threshold for QB/QBx (mm), during '
                        'the first approximation.')

    p.add_argument('--processes', type=int, default=1,
                   help='Number of desired processes [%(default)s].')
    p.add_argument('--convergence', action='store', type=int, default=100,
                   help='Streamlines count difference threshold to stop '
                        're-running the algorithm.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    # Check if the files exist
    if not os.path.isfile(args.input_bundle):
        parser.error('"{0}" must be a file!'.format(args.input_bundle))

    if os.path.isfile(args.output_bundle) and not args.force_overwrite:
        parser.error('"{0}" already exist! Use -f to overwrite it.'.format(
            args.output_bundle))

    min_distance = args.min_distance
    clustering_thr = args.clustering_thr
    nbr_cpu = args.processes

    trk_file = nib.streamlines.load(args.input_bundle)
    streamlines = list(trk_file.streamlines)
    original_length = len(streamlines)
    # print 'Loaded '+str(original_length)+' streamlines...'

    pool = multiprocessing.Pool(nbr_cpu)
    timer = time()
    # print 'Lauching subsampling on '+str(nbr_cpu)+' processes...'
    while True:
        current_iteration_length = len(streamlines)
        skip = int(len(streamlines) / nbr_cpu) + 1

        # Cheap trick to avoid duplication in memory, the pop remove from
        # one list to append it to the other, slower but allows bigger bundles
        if nbr_cpu > 1 and len(streamlines) > 8000:
            split_streamlines_list = []

            for i in range(nbr_cpu):
                split_streamlines_list.append(streamlines[0:skip])
                del streamlines[0:skip]

            resulting_streamlines = pool.map(multiprocess_subsampling,
                                             zip(split_streamlines_list,
                                                 repeat(min_distance),
                                                 repeat(clustering_thr), 
                                                 repeat(1),
                                                 repeat(False)))

            # Fused all subprocesses' result together
            streamlines = list(chain(*resulting_streamlines))

        difference_length = current_iteration_length - len(streamlines)

        # print 'Difference (before - after):', \
            # difference_length, 'streamlines were removed'
        if difference_length < args.convergence:
            # print 'The smart-subsampling converged,' \
                #   '( below '+str(args.convergence)+')'
            break
        else:
            # print 'Threshold of convergence was not achieved, ' \
                #   'Need another run...\n'
            random.shuffle(streamlines)

    # After convergence, we can simply save the output
    # print
    # print 'Before -> After:', str(original_length)+' -> '+str(len(streamlines)), \
        'in {:2f} sec.'.format(time() - timer)
    new_tractogram = nib.streamlines.Tractogram(streamlines,
                                                affine_to_rasmm=np.eye(4))
    nib.streamlines.save(new_tractogram, args.output_bundle,
                         header=trk_file.header)


if __name__ == "__main__":
    main()
