#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute BundleSeg & supports multi-atlas & multi-parameters (RBx-like).
The model needs to be cleaned and lightweight.
Transform should come from ANTs: (using the --inverse flag)
AntsRegistrationSyNQuick.sh -d 3 -m MODEL_REF -f SUBJ_REF

If you are not sure about the transformation 'direction' you can try
scil_tractogram_segment_bundles.py (with the -v option), a warning will popup
if the provided transformation is not used correctly.

The number of folders inside 'models_directories' will increase the number of
runs. Each folder is considered like an atlas and bundles inside will initiate
more BundleSeg executions. The more atlases you have, the more robust the
recognition will be.

--minimal_vote_ratio is a value between 0 and 1. If you have 5 input model
directories and a minimal_vote_ratio of 0.5, you will need at least 3 votes

Example data and usage available at: https://zenodo.org/record/7950602

For RAM usage, it is recommanded to use this heuristic:
    (size of inputs tractogram (GB) * number of processes) < RAM (GB)
This is important because many instances of data structures are initialized
in parallel and can lead to a RAM overflow.

Formerly: scil_recognize_multi_bundles.py
"""

import argparse
import logging
import json
import os

import coloredlogs
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             load_matrix_in_any_format,
                             add_verbose_arg,
                             add_reference_arg)
from scilpy.segment.voting_scheme import VotingScheme

EPILOG = """
[1] St-Onge, Etienne, Kurt G. Schilling, and Francois Rheault.
"BundleSeg: A versatile,reliable and reproducible approach to white
matter bundle segmentation." International Workshop on Computational
Diffusion MRI. Cham: Springer Nature Switzerland (2023)
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
        epilog=EPILOG)

    p.add_argument('in_tractograms', nargs='+',
                   help='Input tractogram filename (.trk or .tck).')
    p.add_argument('in_config_file',
                   help='Path of the config file (.json)')
    p.add_argument('in_directory',
                   help='Path of parent folder of models directories.\n'
                        'Each folder inside will be considered as a'
                        'different atlas.')
    p.add_argument('in_transfo',
                   help='Path for the transformation to model space '
                        '(.txt, .npy or .mat).')

    p.add_argument('--out_dir', default='voting_results',
                   help='Path for the output directory [%(default)s].')
    p.add_argument('--minimal_vote_ratio', type=float, default=0.5,
                   help='Streamlines will only be considered for saving if\n'
                        'recognized often enough [%(default)s].')

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
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    args.in_models_directories = [os.path.join(args.in_directory, x)
                                  for x in os.listdir(args.in_directory)
                                  if os.path.isdir(os.path.join(
                                                    args.in_directory, x))]

    assert_inputs_exist(parser, args.in_tractograms +
                        [args.in_config_file, args.in_transfo],
                        args.reference)

    for in_tractogram in args.in_tractograms:
        ext = os.path.splitext(in_tractogram)[1]
        if ext not in ['.trk', '.trx'] and args.reference is None:
            parser.error('A reference is needed for {} file'.format(ext))

    for directory in args.in_models_directories:
        if not os.path.isdir(directory):
            parser.error('Input folder {0} does not exist'.format(directory))

    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)

    file_handler = logging.FileHandler(filename=os.path.join(args.out_dir,
                                                             'logfile.txt'))
    formatter = logging.Formatter(
        fmt='%(asctime)s, %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    coloredlogs.install(level=logging.getLevelName(args.verbose))

    transfo = load_matrix_in_any_format(args.in_transfo)
    if args.inverse:
        transfo = np.linalg.inv(load_matrix_in_any_format(args.in_transfo))

    with open(args.in_config_file) as json_data:
        config = json.load(json_data)

    # For code simplicity, it is still RecobundlesX class and all, but
    # the last pruning step was modified to be in line with BundleSeg.
    voting = VotingScheme(config, args.in_models_directories,
                          transfo, args.out_dir,
                          minimal_vote_ratio=args.minimal_vote_ratio)

    voting(args.in_tractograms, nbr_processes=args.nbr_processes,
           seed=args.seed, reference=args.reference)


if __name__ == '__main__':
    main()
