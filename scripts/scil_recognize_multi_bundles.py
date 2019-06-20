#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Compute  RecobundlesX (multi-atlas & multi-parameters).
    The model need to be cleaned and lightweight.
    Transform should come from ANTs: (using the --inverse flag)
    AntsRegistration -m MODEL_REF -f SUBJ_REF
    ConvertTransformFile 3 0GenericAffine.mat 0GenericAffine.npy --ras --hm
"""

import argparse
import logging
import json
import os
import random
import shutil

import coloredlogs
import numpy as np

from scilpy.io.utils import add_overwrite_arg, assert_inputs_exist
from scilpy.segment.voting_scheme import VotingScheme


def _buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Input tractogram filename (trk or tck).')
    p.add_argument('config_file',
                   help='Path of the config file (json)')
    p.add_argument('models_directories', nargs='+',
                   help='Path for the directories containing model.')
    p.add_argument('transformation',
                   help='Path for the transformation to model space.')

    p.add_argument('--output_dir', default='voting_results/',
                   help='Path for the output directory.')
    p.add_argument('--log_level', default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                   help='Log level of the logging class')

    p.add_argument('--multi_parameters', type=int, default=1,
                   help='Pick parameters from the potential combinations\n' +
                   'Will multiply the number of time Recobundles is ran.')
    p.add_argument('--minimal_vote_ratio', type=float, default=0.5,
                   help='Streamlines will only be considered for saving if\n ' +
                   'recognized often enough.')

    p.add_argument('--tractogram_clustering_thr',
                   type=int, default=[12], nargs='+',
                   help='Input tractogram clustering thresholds ' +
                   '[%(default)smm].')

    p.add_argument('--processes', type=int, default=1,
                   help='Number of thread used for computation [%(default)s].')
    p.add_argument('--seeds', type=int, default=[None], nargs='+',
                   help='Random number generator seed [%(default)s]\n' +
                   'Will multiply the number of time Recobundles is ran.')
    p.add_argument('--inverse', action='store_true',
                   help='Use the inverse transformation.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram,
                                 args.config_file,
                                 args.transformation])

    for directory in args.models_directories:
        if not os.path.isdir(directory):
            parser.error('Input folder {0} does not exist'.format(directory))

    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        elif args.overwrite:
            shutil.rmtree(args.output_dir)
            os.mkdir(args.output_dir)
        else:
            parser.error('Output folder {0} exists. Use -f to force '
                         'overwriting'.format(args.output_dir))

    logging.basicConfig(filename=os.path.join(args.output_dir, 'logfile.txt'),
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
                          transfo, args.output_dir,
                          minimal_vote_ratio=args.minimal_vote_ratio,
                          multi_parameters=args.multi_parameters)

    if args.seeds is None:
        seeds = [random.randint(1, 1000)]
    else:
        seeds = args.seeds

    voting.multi_recognize(args.in_tractogram, args.tractogram_clustering_thr,
                           nbr_processes=args.processes, seeds=seeds)


if __name__ == '__main__':
    main()
