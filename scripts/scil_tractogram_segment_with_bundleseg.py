#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute BundleSeg & supports multi-atlas & multi-parameters (RBx-like).

For a single bundle segmentation, see the lighter version:
>>> scil_tractogram_segment_with_recobundles.py

Hints:
- The model needs to be cleaned and lightweight.
- The transform should come from ANTs: (using the --inverse flag)
  >>> AntsRegistrationSyNQuick.sh -d 3 -m MODEL_REF -f SUBJ_REF
  If you are not sure about the transformation 'direction' you can try
scil_tractogram_segment_with_recobundles.py. See its documentation for
explanation on how to verify the direction.
- The number of folders inside 'models_directories' will increase the number of
runs. Each folder is considered like an atlas and bundles inside will initiate
more BundleSeg executions. The more atlases you have, the more robust the
recognition will be.

Example data and usage available at: https://zenodo.org/record/7950602

For CPU usage, it can be variable (advanced CPU vs. basic CPU):
    On personal computer: 4 CPU per subject and then it is better to parallelize
    across subjects.
    On a cluster: 8 CPU per subject and then it is better to parallelize across
    subjects.

For RAM usage, it is recommanded to use this heuristic:
    (size of inputs tractogram (GB) * number of processes) < RAM (GB)
This is important because many instances of data structures are initialized
in parallel and can lead to a RAM overflow.

Formerly: scil_recognize_multi_bundles.py
------------------------------------------------------------------------------------------
Reference:
[1] St-Onge, Etienne, Kurt G. Schilling, and Francois Rheault."BundleSeg: A versatile,
    reliable and reproducible approach to white matter bundle segmentation." International 
    Workshop on Computational Diffusion MRI. Cham: Springer Nature Switzerland (2023)
------------------------------------------------------------------------------------------
"""

import argparse
import logging
import json
import os

import coloredlogs
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_reference_arg, add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             load_matrix_in_any_format, ranged_type)
from scilpy.segment.voting_scheme import VotingScheme
from scilpy.version import version_string

logger = logging.getLogger('BundleSeg')

EPILOG = """
[1] St-Onge, Etienne, Kurt G. Schilling, and Francois Rheault.
"BundleSeg: A versatile,reliable and reproducible approach to white
matter bundle segmentation." International Workshop on Computational
Diffusion MRI. Cham: Springer Nature Switzerland (2023)
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractograms', nargs='+',
                   help='Input tractogram filename (.trk or .tck).')
    p.add_argument('in_config_file',
                   help='Path of the config file (.json)')
    p.add_argument('in_directory',
                   help='Path of parent folder of models directories.\n'
                        'Each folder inside will be considered as a '
                        'different atlas.')
    p.add_argument('in_transfo',
                   help='Path for the transformation to model space '
                        '(.txt, .npy or .mat).')

    p.add_argument('--out_dir', default='voting_results',
                   help='Path for the output directory [%(default)s].')
    p.add_argument('--minimal_vote_ratio',
                   type=ranged_type(float, 0, 1), default=0.5,
                   help='Streamlines will only be considered for saving if\n'
                        'recognized often enough.\n'
                        'The ratio is a value between 0 and 1. Ex: If you '
                        'have 5 input model directories and a '
                        'minimal_vote_ratio of 0.5, you will need at least 3 '
                        'votes. [%(default)s]')

    g = p.add_argument_group(title='Exploration mode')
    p2 = g.add_mutually_exclusive_group()
    p2.add_argument('--exploration_mode', action='store_true',
                    help='Use higher pruning threshold, but optimal filtering '
                    'can be explored using scil_bundle_explore_bundleseg.py')
    p2.add_argument('--modify_distance_thr', type=float, default=0.0,
                    help='Increase or decrease the distance threshold for '
                         'pruning for all bundles in the configuration '
                         '[%(default)s]')

    p.add_argument('--seed', type=int, default=0,
                   help='Random number generator seed %(default)s.')
    p.add_argument('--inverse', action='store_true',
                   help='Use the inverse transformation.')

    add_reference_arg(p)
    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logger.setLevel(logging.getLevelName(args.verbose))
    logging.getLogger().setLevel(logging.getLevelName('INFO'))

    # Verifications
    in_models_directories = [
        os.path.join(args.in_directory, x)
        for x in os.listdir(args.in_directory)
        if os.path.isdir(os.path.join(args.in_directory, x))]
    if len(in_models_directories) == 0:
        parser.error("Found no model in {}".format(args.in_directory))
    logger.info("Found {} models in your model directory!"
                .format(len(in_models_directories)))

    assert_inputs_exist(parser, args.in_tractograms +
                        [args.in_config_file, args.in_transfo],
                        args.reference)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)

    # Loading
    transfo = load_matrix_in_any_format(args.in_transfo)

    with open(args.in_config_file) as json_data:
        config = json.load(json_data)

    if args.exploration_mode:
        for key in config.keys():
            config[key] = 10
    elif args.modify_distance_thr is not None:
        for key in config.keys():
            config[key] += args.modify_distance_thr

    # (verifying now tractograms' extensions. Loading will only be later.)
    for in_tractogram in args.in_tractograms:
        ext = os.path.splitext(in_tractogram)[1]
        if ext not in ['.trk', '.trx'] and args.reference is None:
            parser.error('A reference is needed for {} files'.format(ext))

    # Managing the logging
    file_handler = logging.FileHandler(
        filename=os.path.join(args.out_dir, 'logfile.txt'))
    formatter = logging.Formatter(
        fmt='%(asctime)s, %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    coloredlogs.install(level=logging.getLevelName(args.verbose))

    # Processing.
    if args.inverse:
        transfo = np.linalg.inv(transfo)

    # Note. Loading and saving are managed through the VotingScheme class.
    # For code simplicity, it is still BundleSeg class and all, but
    # the last pruning step was modified to be in line with BundleSeg.

    voting = VotingScheme(config, in_models_directories,
                          transfo, args.out_dir,
                          minimal_vote_ratio=args.minimal_vote_ratio)

    voting(args.in_tractograms, nbr_processes=args.nbr_processes,
           seed=args.seed, reference=args.reference)


if __name__ == '__main__':
    main()
