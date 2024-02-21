#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs a network-based statistical comparison for populations g1 and g2. The
output is a matrix of the same size as the input connectivity matrices, with
p-values at each edge.
All input matrices must have the same shape (NxN). For paired t-test, both
groups must have the same number of observations.

For example, if you have streamline count weighted matrices for a MCI and a
control group and you want to investiguate differences in their connectomes:
   >>> scil_connectivity_compare_populations.py pval.npy
           --g1 MCI/*_sc.npy --g2 CTL/*_sc.npy

--filtering_mask will simply multiply the binary mask to all input
matrices before performing the statistical comparison. Reduces the number of
statistical tests, useful when using --fdr or --bonferroni.

Formerly: scil_compare_connectivity.py
"""

import argparse
import logging

import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format)
from scilpy.stats.matrix_stats import ttest_two_matrices

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
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EPILOG)

    p.add_argument('out_pval_matrix',
                   help='Output matrix (.npy) containing the edges p-value.')

    p.add_argument('--in_g1', nargs='+', required=True,
                   help='List of matrices for the first population (.npy).')
    p.add_argument('--in_g2', nargs='+', required=True,
                   help='List of matrices for the second population (.npy).')
    p.add_argument('--tail', choices=['left', 'right', 'both'], default='both',
                   help='Enables specification of an alternative hypothesis:\n'
                        'left: mean of g1 < mean of g2,\n'
                        'right: mean of g2 < mean of g1,\n'
                        'both: both means are not equal (default).')
    p.add_argument('--paired', action='store_true',
                   help='Use paired sample t-test instead of population t-test'
                        '.\n--in_g1 and --in_g2 must be ordered the same way.')

    fwe = p.add_mutually_exclusive_group()
    fwe.add_argument('--fdr', action='store_true',
                     help='Perform a false discovery rate (FDR) correction '
                          'for the p-values.\nUses the number of non-zero '
                          'edges as number of tests (value between 0.01 and '
                          '0.1).')
    fwe.add_argument('--bonferroni', action='store_true',
                     help='Perform a Bonferroni correction for the p-values.\n'
                          'Uses the number of non-zero edges as number of '
                          'tests.')

    p.add_argument('--p_threshold', nargs=2, metavar=('THRESH', 'OUT_FILE'),
                   help='Threshold the final p-value matrix and save the '
                        'binary matrix (.npy).')
    p.add_argument('--filtering_mask',
                   help='Binary filtering mask (.npy) to apply before '
                        'computing the measures.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_g1+args.in_g2,
                        args.filtering_mask)
    assert_outputs_exist(parser, args, args.out_pval_matrix)

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

    if matrices_g1.shape[0:2] != matrices_g2.shape[0:2]:
        parser.error('Both groups have different matrices dimensions (NxN).')
    if args.paired and matrices_g1.shape[2] != matrices_g2.shape[2]:
        parser.error('For paired statistic both groups must have the same '
                     'number of observations.')

    matrix_pval = ttest_two_matrices(matrices_g1, matrices_g2, args.paired,
                                     args.tail, args.fdr, args.bonferroni)

    save_matrix_in_any_format(args.out_pval_matrix, matrix_pval)

    # Save the significant edges (equivalent to an upper_threshold)
    # 0 where it is not significant and 1 where it is significant
    if args.p_threshold:
        p_thresh = float(args.p_threshold[0])
        matrix_shape = matrices_g1.shape[0:2]
        masked_pval_matrix = np.zeros(matrix_shape)
        logging.info('Threshold the p-values at {}'.format(p_thresh))
        masked_pval_matrix[matrix_pval < p_thresh] = 1
        masked_pval_matrix[matrix_pval < 0] = 0

        save_matrix_in_any_format(args.p_threshold[1], masked_pval_matrix)


if __name__ == '__main__':
    main()
