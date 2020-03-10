#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute RecobundlesX (multi-atlas & multi-parameters).
The model needs to be cleaned and lightweight.
Transform should come from ANTs: (using the --inverse flag)
AntsRegistration -m MODEL_REF -f SUBJ_REF
ConvertTransformFile 3 0GenericAffine.mat 0GenericAffine.npy --ras --hm

The next two arguments are multi-parameters related:
--multi_parameters must be lower than len(model_clustering_thr) *
len(bundle_pruning_thr) * len(tractogram_clustering_thr)

--seeds can be more than one value. Multiple values will result in
a overall multiplicative factor of len(seeds) * '--multi_parameters'

The number of folder provided by 'models_directories' will further multiply
the total number of run. Meaning that the total number of Recobundle
execution will be len(seeds) * '--multi_parameters' * len(models_directories)

--minimal_vote_ratio is a value between 0 and 1. The actual number of vote
required will be '--minimal_vote_ratio' * len(seeds) * '--multi_parameters'
* len(models_directories).

Example: 5 atlas, 9 multi-parameters, 2 seeds with a minimal vote_ratio
of 0.50 will results in 90 executions (for each bundle in the config file)
and a minimal vote of 45 / 90.

Example data and usage available at: https://zenodo.org/deposit/3613688
"""

import argparse
import logging
import json
import os
import random

import coloredlogs
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.segment.voting_scheme import VotingScheme


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
        epilog="""Garyfallidis, E., Côté, M. A., Rheault, F., ... &
        Descoteaux, M. (2018). Recognition of white matter
        bundles using local and global streamline-based registration and
        clustering. NeuroImage, 170, 283-295.""")

    p.add_argument('in_tractogram',
                   help='Input tractogram filename (trk or tck).')
    p.add_argument('config_file',
                   help='Path of the config file (json)')
    p.add_argument('models_directories', nargs='+',
                   help='Path for the directories containing model.')
    p.add_argument('transformation',
                   help='Path for the transformation to model space.')

    p.add_argument('--output', default='voting_results/',
                   help='Path for the output directory [%(default)s].')
    p.add_argument('--log_level', default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                   help='Log level of the logging class.')

    p.add_argument('--multi_parameters', type=int, default=1,
                   help='Pick parameters from the potential combinations\n'
                        'Will multiply the number of time Recobundles is ran.\n'
                        'See the documentation [%(default)s].')
    p.add_argument('--minimal_vote_ratio', type=float, default=0.5,
                   help='Streamlines will only be considered for saving if\n'
                        'recognized often enough [%(default)s].')

    p.add_argument('--tractogram_clustering_thr',
                   type=int, default=[12], nargs='+',
                   help='Input tractogram clustering thresholds %(default)smm.')

    p.add_argument('--processes', type=int, default=1,
                   help='Number of thread used for computation [%(default)s].')
    p.add_argument('--seeds', type=int, default=[None], nargs='+',
                   help='Random number generator seed %(default)s\n'
                        'Will multiply the number of time Recobundles is ran.')
    p.add_argument('--inverse', action='store_true',
                   help='Use the inverse transformation.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram,
                                 args.config_file,
                                 args.transformation])

    for directory in args.models_directories:
        if not os.path.isdir(directory):
            parser.error('Input folder {0} does not exist'.format(directory))

    assert_output_dirs_exist_and_empty(parser, args, args.output)

    logging.basicConfig(filename=os.path.join(args.output, 'logfile.txt'),
                        filemode='w',
                        format='%(asctime)s, %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S', level=args.log_level)

    coloredlogs.install(level=args.log_level)

    transfo = np.loadtxt(args.transformation)
    if args.inverse:
        transfo = np.linalg.inv(np.loadtxt(args.transformation))

    with open(args.config_file) as json_data:
        config = json.load(json_data)

    voting = VotingScheme(config, args.models_directories,
                          transfo, args.output,
                          tractogram_clustering_thr=args.tractogram_clustering_thr,
                          minimal_vote_ratio=args.minimal_vote_ratio,
                          multi_parameters=args.multi_parameters)

    if args.seeds is None:
        seeds = [random.randint(1, 1000)]
    else:
        seeds = args.seeds

    voting(args.in_tractogram,
           nbr_processes=args.processes, seeds=seeds)


if __name__ == '__main__':
    main()
