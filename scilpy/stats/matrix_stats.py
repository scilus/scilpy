# -*- coding: utf-8 -*-
import logging

import bct

import numpy as np
from scipy.stats import t as stats_t
from statsmodels.stats.multitest import multipletests


def _ttest_stat_only(x, y, tail):
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


def _ttest_paired_stat_only(x, y, tail):
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


def ttest_two_matrices(matrices_g1, matrices_g2, paired, tail, fdr,
                       bonferroni):
    """
    Parameters
    ----------
    matrices_g1: np.ndarray of shape ?  (toDO)
    matrices_g2: np.ndarray of shape ?
    paired: bool
        Use paired sample t-test instead of population t-test. The two matrices
        must be ordered the same way.
    tail: str.
        One of ['left', 'right', 'both'].
    fdr: bool
        Perform a false discovery rate (FDR) correction for the p-values. Uses
        the number of non-zero edges as number of tests (value between 0.01 and
        0.1).
    bonferroni: bool
        Perform a Bonferroni correction for the p-values. Uses the number of
        non-zero edges as number of tests.
    """
    matrix_shape = matrices_g1.shape[0:2]
    nb_group_g1 = matrices_g1.shape[2]
    nb_group_g2 = matrices_g2.shape[2]

    # Todo better reshape, more simple
    sum_both_groups = np.sum(matrices_g1, axis=2) + np.sum(matrices_g2, axis=2)
    nbr_non_zeros = np.count_nonzero(np.triu(sum_both_groups))

    logging.info('The provided matrices contain {} non zeros elements.'
                 .format(nbr_non_zeros))

    matrices_g1 = matrices_g1.reshape((np.prod(matrix_shape), nb_group_g1))
    matrices_g2 = matrices_g2.reshape((np.prod(matrix_shape), nb_group_g2))
    # Negative epsilon, to differentiate from null p-values
    matrix_pval = np.ones(np.prod(matrix_shape)) * -0.000001

    text = ' paired' if paired else ''
    logging.info('Performing{} t-test with "{}" hypothesis.'
                 .format(text, tail))
    logging.info('Data has dimensions {}x{} with {} and {} observations.'
                 .format(matrix_shape[0], matrix_shape[1],
                          nb_group_g1, nb_group_g2))

    # For conversion to p-values
    if paired:
        dof = nb_group_g1 - 1
    else:
        dof = nb_group_g1 + nb_group_g2 - 2

    for ind in range(np.prod(matrix_shape)):
        # Skip edges with no data, leaves a negative epsilon instead
        if not matrices_g1[ind].any() and not matrices_g2[ind].any():
            continue

        if paired:
            t_stat = (_ttest_paired_stat_only(
                matrices_g1[ind], matrices_g2[ind], tail))
        else:
            t_stat = _ttest_stat_only(
                matrices_g1[ind], matrices_g2[ind], tail)

        pval = stats_t.sf(t_stat, dof)
        matrix_pval[ind] = pval if tail == 'both' else pval / 2.0

    corr_matrix_pval = matrix_pval.reshape(matrix_shape)
    if fdr:
        logging.info('Using FDR, the results will be q-values.')
        corr_matrix_pval = np.triu(corr_matrix_pval)
        corr_matrix_pval[corr_matrix_pval > 0] = multipletests(
            corr_matrix_pval[corr_matrix_pval > 0], 0, method='fdr_bh')[1]

        # Symmetrize  the matrix
        matrix_pval = corr_matrix_pval + corr_matrix_pval.T - \
            np.diag(corr_matrix_pval.diagonal())
    elif bonferroni:
        corr_matrix_pval = np.triu(corr_matrix_pval)
        corr_matrix_pval[corr_matrix_pval > 0] = multipletests(
            corr_matrix_pval[corr_matrix_pval > 0], 0, method='bonferroni')[1]

        # Symmetrize  the matrix
        matrix_pval = corr_matrix_pval + corr_matrix_pval.T - \
            np.diag(corr_matrix_pval.diagonal())
    else:
        matrix_pval = matrix_pval.reshape(matrix_shape)

    return matrix_pval


def omega_sigma(matrix):
    """Returns the small-world coefficients (omega & sigma) of a graph.
    Omega ranges between -1 and 1. Values close to 0 mean the matrix
    features small-world characteristics.
    Values close to -1 mean the network has a lattice structure and values
    close to 1 mean G is a random network.

    A network is commonly classified as small-world if sigma > 1.

    Parameters
    ----------
    matrix : numpy.ndarray
        A weighted undirected graph.
    Returns
    -------
    smallworld : tuple of float
        The small-work coefficients (omega & sigma).
    Notes
    -----
    The implementation is adapted from the algorithm by Telesford et al. [1]_.
    References
    ----------
    .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
           "The Ubiquity of Small-World Networks".
           Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.
           doi:10.1089/brain.2011.0038.
    """
    transitivity_rand_list = []
    transitivity_latt_list = []
    path_length_rand_list = []
    for i in range(10):
        logging.info('Generating random and lattice matrices, '
                     'iteration #{}.'.format(i))
        random = bct.randmio_und(matrix, 10)[0]
        lattice = bct.latmio_und(matrix, 10)[1]

        transitivity_rand_list.append(bct.transitivity_wu(random))
        transitivity_latt_list.append(bct.transitivity_wu(lattice))
        path_length_rand_list.append(
            float(np.average(bct.distance_wei(random)[0])))

    transitivity = bct.transitivity_wu(matrix)
    path_length = float(np.average(bct.distance_wei(matrix)[0]))
    transitivity_rand = np.mean(transitivity_rand_list)
    transitivity_latt = np.mean(transitivity_latt_list)
    path_length_rand = np.mean(path_length_rand_list)

    omega = (path_length_rand / path_length) - \
        (transitivity / transitivity_latt)
    sigma = (transitivity / transitivity_rand) / \
        (path_length / path_length_rand)

    return float(omega), float(sigma)
