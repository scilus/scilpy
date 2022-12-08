#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute RecobundlesX (multi-atlas & multi-parameters).
The model needs to be cleaned and lightweight.
Transform should come from ANTs: (using the --inverse flag)
AntsRegistrationSyNQuick.sh -d 3 -m MODEL_REF -f SUBJ_REF

If you are not sure about the transformation 'direction' you can try
scil_recognize_single_bundle.py (with the -v option), a warning will popup if
the provided transformation is not use correctly.

The next two arguments are multi-parameters related:
--multi_parameters must be lower than len(model_clustering_thr) *
len(bundle_pruning_thr) * len(tractogram_clustering_thr)

--seeds can be more than one value. Multiple values will result in
a overall multiplicative factor of len(seeds) * '--multi_parameters'

The number of folders provided by 'models_directories' will further multiply
the total number of runs. Meaning that the total number of Recobundles
execution will be len(seeds) * '--multi_parameters' * len(models_directories)

--minimal_vote_ratio is a value between 0 and 1. The actual number of votes
required will be '--minimal_vote_ratio' * len(seeds) * '--multi_parameters'
* len(models_directories).

Example: 5 atlas, 9 multi-parameters, 2 seeds with a minimal vote_ratio
of 0.50 will results in 90 executions (for each bundle in the config file)
and a minimal vote of 45 / 90.

Example data and usage available at: https://zenodo.org/record/3928503
"""

import argparse
import logging
import json
import os
import random

import coloredlogs
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             load_matrix_in_any_format)
from scilpy.segment.voting_scheme import VotingScheme

EPILOG = """
Garyfallidis, E., Cote, M. A., Rheault, F., ... &
Descoteaux, M. (2018). Recognition of white matter
bundles using local and global streamline-based registration and
clustering. NeuroImage, 170, 283-295.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
        epilog=EPILOG)

    p.add_argument('in_tractogram',
                   help='Input tractogram filename (.trk or .tck).')
    p.add_argument('in_config_file',
                   help='Path of the config file (.json)')
    p.add_argument('in_models_directories', nargs='+',
                   help='Path for the directories containing model.')
    p.add_argument('in_transfo',
                   help='Path for the transformation to model space '
                        '(.txt, .npy or .mat).')

    p.add_argument('--out_dir', default='voting_results',
                   help='Path for the output directory [%(default)s].')
    p.add_argument('--log_level', default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                   help='Log level of the logging class.')

    p.add_argument('--multi_parameters', type=int, default=1,
                   help='Pick parameters from the potential combinations\n'
                        'Will multiply the number of times Recobundles is ran.\n'
                        'See the documentation [%(default)s].')
    p.add_argument('--minimal_vote_ratio', type=float, default=0.5,
                   help='Streamlines will only be considered for saving if\n'
                        'recognized often enough [%(default)s].')

    p.add_argument('--tractogram_clustering_thr',
                   type=int, default=[12], nargs='+',
                   help='Input tractogram clustering thresholds %(default)smm.')

    p.add_argument('--seeds', type=int, default=[0], nargs='+',
                   help='Random number generator seed %(default)s\n'
                        'Will multiply the number of times Recobundles is ran.')
    p.add_argument('--inverse', action='store_true',
                   help='Use the inverse transformation.')

    add_processes_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram,
                                 args.in_config_file,
                                 args.in_transfo])

    for directory in args.in_models_directories:
        if not os.path.isdir(directory):
            parser.error('Input folder {0} does not exist'.format(directory))

    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)

    file_handler = logging.FileHandler(filename=os.path.join(args.out_dir,
                                                             'logfile.txt'))
    formatter = logging.Formatter(fmt='%(asctime)s, %(name)s %(levelname)s %(message)s',
                                  datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    logging.getLogger().setLevel(args.log_level)
    logging.getLogger().addHandler(file_handler)
    coloredlogs.install(level=args.log_level)

    transfo = load_matrix_in_any_format(args.in_transfo)
    if args.inverse:
        transfo = np.linalg.inv(load_matrix_in_any_format(args.in_transfo))

    with open(args.in_config_file) as json_data:
        config = json.load(json_data)

    voting = VotingScheme(config, args.in_models_directories,
                          transfo, args.out_dir,
                          tractogram_clustering_thr=args.tractogram_clustering_thr,
                          minimal_vote_ratio=args.minimal_vote_ratio,
                          multi_parameters=args.multi_parameters)

    if args.seeds is None:
        seeds = [random.randint(1, 1000)]
    else:
        seeds = args.seeds

    voting(args.in_tractogram, nbr_processes=args.nbr_processes, seeds=seeds)


if __name__ == '__main__':
    main()
