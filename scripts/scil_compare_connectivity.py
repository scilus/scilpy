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
>>> scil_compare_connectivity.py pval.npy --g1 MCI/*_sc.npy --g2 CTL/*_sc.npy

--filtering_mask will simply multiply the binary mask to all input
matrices before performing the statistical comparison. Reduces the number of
statistical tests, useful when using --fdr or --bonferroni.
"""

import argparse
import logging

import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
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
                   help='Use paired sample t-test instead of population t-test.\n'
                        '--in_g1 and --in_g2 must be ordered the same way.')

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


def ttest_stat_only(x, y, tail):
    t = np.mean(x) - np.mean(y)
    n1, n2 = len(x), len(y)
    s = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1)
                 * np.var(y, ddof=1)) / (n1 + n2 - 2))
    denom = s * np.sqrt(1 / n1 + 1 / n2)
    if denom == 0:
        return 0
    if tail == 'both':
        return np.abs(t / denom)
    if tail == 'left':
        return -t / denom
    else:
        return t / denom


def ttest_paired_stat_only(x, y, tail):
    n = len(x - y)
    sample_ss = np.sum((x - y)**2) - np.sum(x - y)**2 / n
    unbiased_std = np.sqrt(sample_ss / (n - 1))

    z = np.mean(x - y) / unbiased_std
    t = z * np.sqrt(n)
    if tail == 'both':
        return np.abs(t)
    if tail == 'left':
        return -t
    else:
        return t


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_g1+args.in_g2,
                        args.filtering_mask)
    assert_outputs_exist(parser, args, args.out_pval_matrix)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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

    matrix_shape = matrices_g1.shape[0:2]
    nb_group_g1 = matrices_g1.shape[2]
    nb_group_g2 = matrices_g2.shape[2]

    # To do better reshape, more simple
    sum_both_groups = np.sum(matrices_g1, axis=2) + np.sum(matrices_g2, axis=2)
    nbr_non_zeros = np.count_nonzero(np.triu(sum_both_groups))

    logging.debug('The provided matrices contain {} non zeros elements.'.format(
        nbr_non_zeros))

    matrices_g1 = matrices_g1.reshape((np.prod(matrix_shape), nb_group_g1))
    matrices_g2 = matrices_g2.reshape((np.prod(matrix_shape), nb_group_g2))
    # Negative epsilon, to differenciate from null p-values
    matrix_pval = np.ones(np.prod(matrix_shape)) * -0.000001

    text = ' paired' if args.paired else ''
    logging.debug('Performing{} t-test with "{}" hypothesis.'.format(text,
                                                                     args.tail))
    logging.debug('Data has dimensions {}x{} with {} and {} observations.'.format(
        matrix_shape[0], matrix_shape[1],
        nb_group_g1, nb_group_g2))

    # For conversion to p-values
    if args.paired:
        dof = nb_group_g1 - 1
    else:
        dof = nb_group_g1 + nb_group_g2 - 2

    for ind in range(np.prod(matrix_shape)):
        # Skip edges with no data, leaves a negative epsilon instead
        if not matrices_g1[ind].any() and not matrices_g2[ind].any():
            continue

        if args.paired:
            t_stat = ttest_paired_stat_only(matrices_g1[ind], matrices_g2[ind],
                                            args.tail)
        else:
            t_stat = ttest_stat_only(matrices_g1[ind], matrices_g2[ind],
                                     args.tail)

        pval = t.sf(t_stat, dof)
        matrix_pval[ind] = pval if args.tail == 'both' else pval / 2.0

    corr_matrix_pval = matrix_pval.reshape(matrix_shape)
    if args.fdr:
        logging.debug('Using FDR, the results will be q-values.')
        corr_matrix_pval = np.triu(corr_matrix_pval)
        corr_matrix_pval[corr_matrix_pval > 0] = multipletests(
            corr_matrix_pval[corr_matrix_pval > 0], 0, method='fdr_bh')[1]

        # Symmetrize  the matrix
        matrix_pval = corr_matrix_pval + corr_matrix_pval.T - \
            np.diag(corr_matrix_pval.diagonal())
    elif args.bonferroni:
        corr_matrix_pval = np.triu(corr_matrix_pval)
        corr_matrix_pval[corr_matrix_pval > 0] = multipletests(
            corr_matrix_pval[corr_matrix_pval > 0], 0, method='bonferroni')[1]

        # Symmetrize  the matrix
        matrix_pval = corr_matrix_pval + corr_matrix_pval.T - \
            np.diag(corr_matrix_pval.diagonal())
    else:
        matrix_pval = matrix_pval.reshape(matrix_shape)

    save_matrix_in_any_format(args.out_pval_matrix, matrix_pval)

    # Save the significant edges (equivalent to an upper_threshold)
    # 0 where it is not significant and 1 where it is significant
    if args.p_threshold:
        p_thresh = float(args.p_threshold[0])
        masked_pval_matrix = np.zeros(matrix_shape)
        logging.debug('Threshold the p-values at {}'.format(p_thresh))
        masked_pval_matrix[matrix_pval < p_thresh] = 1
        masked_pval_matrix[matrix_pval < 0] = 0

        save_matrix_in_any_format(args.p_threshold[1], masked_pval_matrix)


if __name__ == '__main__':
    main()
