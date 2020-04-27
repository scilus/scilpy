#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performs a network-based statistical comparison for populations G1 and G2 using
a t-statistic threshold of alpha. All matrices must have the same shape.

For example, if you have streamline count weighted matrices for a MCI and a
control group and you want to investiguate difference in their connectomme:
    scil_compare_connectivity.py --g1 MCI/*_sc.npy --g2 CTL/*_sc.npy

--filtering_mask will simply multiply the binary mask to with all input
matrices before performing the statistical compariso.

For more details visit (notes after the docstring):
https://github.com/aestrivex/bctpy/wiki#network-based-statistic
"""

import argparse
import logging
import os

import bct
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format)


EPILOG = """
[1] Rubinov, Mikail, and Olaf Sporns. "Complex network measures of brain
    connectivity: uses and interpretations." Neuroimage 52.3 (2010):
    1059-1069.
[2] Zalesky, Andrew, Alex Fornito, and Edward T. Bullmore. "Network-based
    statistic: identifying differences in brain networks." Neuroimage 53.4
    (2010): 1197-1207.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=EPILOG)

    p.add_argument('out_matrix',
                   help='Output matrix (.npy) containing the edges p-value.')

    p.add_argument('--in_g1', nargs='+', required=True,
                   help='List of matrices for the first population (.npy).')
    p.add_argument('--in_g2', nargs='+', required=True,
                   help='List of matrices for the second population (.npy).')

    p.add_argument('--t_value', type=float, default=3,
                   help='Minimum t-value used as threshold [%(default)s].')
    p.add_argument('--nb_permutations', type=int, default=1000,
                   help='Number of permutations used to estimate the empirical '
                        'null distribution [%(default)s].')
    p.add_argument('--tail', choices=['left', 'right', 'both'], default='both',
                   help='Enables specification of an alternative hypothesis:\n'
                        'left: mean of g1 < mean of g2,\n'
                        'right: mean of g2 < mean of g1,\n'
                        'both: both means are unequal (default).')
    p.add_argument('--paired', action='store_true',
                   help='Use paired sample t-test instead of population t-test.\n'
                        '--in_g1 and --in_g2 must be ordered the same way.')
    p.add_argument('--filtering_mask',
                   help='Binary filtering mask (.npy) to apply before computing the '
                        'measures.')

    p.add_argument('--save_all', metavar='OUT_DIR',
                   help='Save all the subcomponents in a text format in the'
                        'specified output directory.')
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_g1+args.in_g2,
                        args.filtering_mask)
    assert_outputs_exist(parser, args, args.out_matrix)

    if args.save_all:
        assert_output_dirs_exist_and_empty(parser, args, args.save_all,
                                           create_dir=True)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.filtering_mask:
        filtering_mask = load_matrix_in_any_format(args.filtering_mask)
    else:
        filtering_mask = 1

    matrices_g1 = [load_matrix_in_any_format(i)*filtering_mask
                   for i in args.in_g1]
    matrices_g2 = [load_matrix_in_any_format(i)*filtering_mask
                   for i in args.in_g2]
    matrices_g1 = np.rollaxis(np.array(matrices_g1),
                              axis=0, start=3)
    matrices_g2 = np.rollaxis(np.array(matrices_g2),
                              axis=0, start=3)
    pval, adj, _ = bct.nbs.nbs_bct(matrices_g1, matrices_g2,
                                   thresh=args.t_value,
                                   k=args.nb_permutations,
                                   tail=args.tail,
                                   paired=args.paired,
                                   verbose=args.verbose)
    logging.debug(
        '{} components has been found in the networks'.format(len(pval)))
    if args.save_all:
        for i in range(len(pval)):
            coord = np.array(np.where(adj == i)).T
            results = np.zeros((len(coord), 3))
            results[:, 0:2] = coord
            results[:, 2] = pval[i]
            filename = os.path.join(args.save_all,
                                    'component_{}.txt'.format(i))
            # Manual array saving to support int for coord and float for pval
            with open(filename, 'a') as the_file:
                for j in range(len(results)):
                    the_file.write('{} {} {}\n'.format(int(results[j, 0]),
                                                     int(results[j, 1]),
                                                     round(results[j, 2], 5)))
            logging.debug('Components #{} had {} elements and a pvalue of {}'.format(
                i, len(coord), pval[i]))

    matrix = np.zeros(adj.shape)
    for i in range(len(pval)):
        matrix[adj == i] = pval[i]

    save_matrix_in_any_format(args.out_matrix, matrix)


if __name__ == '__main__':
    main()
