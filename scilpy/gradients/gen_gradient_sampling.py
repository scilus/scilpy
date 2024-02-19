# -*- coding: utf-8 -*-

"""
Most of this code was modified from code by Emmanuel Caruyer
<caruyer@gmail.com>.

See his original code on GitHub:
https://github.com/ecaruyer/qspace/tree/master

The code was reorganized, but general process is kept the same.
"""

import numpy as np
from scipy import optimize

from scilpy.gradients.utils import random_uniform_on_sphere


def generate_gradient_sampling(nb_samples_per_shell, verbose=1):
    """
    Wrapper code to generate gradient sampling from Caruyer's
    multiple_shell_energy.py

    Code to generate multiple-shell gradient sampling, with optimal angular
    coverage. This implements the method described in Caruyer et al.,
    MRM 69(6), pp. 1534-1540, 2013.

    Generates the bvecs of a multiple shell gradient sampling using generalized
    Jones electrostatic repulsion.

    Parameters
    ----------
    nb_samples_per_shell: list[int]
        Number of samples for each shell, starting from lowest.
    verbose: int
        0 = silent, 1 = summary upon completion, 2 = print iterations
        (To be sent to scipy).

    Return
    ------
    bvecs: numpy.array of shape [n, 3]
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for each bvec.
    """

    nb_shells = len(nb_samples_per_shell)

    # Groups of shells and relative coupling weights
    shell_groups = ()
    for i in range(nb_shells):
        shell_groups += ([i],)

    shell_groups += (list(range(nb_shells)),)
    alphas = list(len(shell_groups) * (1.0,))
    weights = _compute_weights(nb_shells, nb_samples_per_shell,
                               shell_groups, alphas)

    # Where the optimized gradient sampling is computed
    # max_iter hardcoded to fit default Caruyer's value
    bvecs = _generate_gradient_sampling_with_weights(
        nb_shells, nb_samples_per_shell, weights, max_iter=100,
        verbose=verbose)

    shell_idx = np.repeat(range(nb_shells), nb_samples_per_shell)

    return bvecs, shell_idx


def _compute_weights(nb_shells, nb_points_per_shell, shell_groups, alphas):
    """
    Computes the weights array from a set of shell groups to couple, and
    coupling weights.

    Parameters
    ----------
    nb_shells: int
        Number of shells
    nb_points_per_shell: list of ints
        Number of points per shell
    shell_groups: tuple
        tuple listing the groups of shells as lists of indices
    alphas: list
        list of weights per group of shells

    Returns
    -------
    weights: nd.ndarray
        weigths for each group of shells
    """
    weights = np.zeros((nb_shells, nb_shells))
    for shell_group, alpha in zip(shell_groups, alphas):
        total_nb_points = 0
        for shell_id in shell_group:
            total_nb_points += nb_points_per_shell[shell_id]
        for i in shell_group:
            for j in shell_group:
                weights[i, j] += alpha / total_nb_points**2
    return weights


def _generate_gradient_sampling_with_weights(
        nb_shells, nb_points_per_shell, weights, max_iter=100, verbose=2):
    """
    Creates a set of sampling directions on the desired number of shells.

    Parameters
    ----------
    nb_shells : int
        The number of shells
    nb_points_per_shell : list, shape (nb_shells,)
        A list of integers containing the number of points on each shell.
    weights : array-like, shape (S, S)
        weighting parameter, control coupling between shells and how this
        balances.
    max_iter: int
        Maximum number of interations

    Returns
    -------
    bvecs : array shape (K, 3) where K is the total number of points
            The points are stored by shell.
    """
    # Total number of points
    nb_point_total = np.sum(nb_points_per_shell)

    # Initialized with random directions
    bvecs = random_uniform_on_sphere(nb_point_total)
    bvecs = bvecs.reshape(nb_point_total * 3)

    bvecs = optimize.fmin_slsqp(_multiple_shell_energy, bvecs,
                                f_eqcons=_constraint_is_bvec_on_sphere,
                                fprime=_grad_multiple_shell_energy,
                                iter=max_iter,
                                acc=1.0e-9,
                                args=(nb_shells, nb_points_per_shell, weights),
                                iprint=verbose)
    bvecs = bvecs.reshape((nb_point_total, 3))
    bvecs = (bvecs.T / np.sqrt((bvecs ** 2).sum(1))).T
    return bvecs


def _multiple_shell_energy(bvecs, nb_shells, nb_points_per_shell, weights):
    """
    Objective function (cost function) for multiple-shell energy.

    This is the main function called during optimization, used as
    func(x, *args) with args = (nb_shells, nb_points_per_shell, weights)

    Parameters
    ----------
    bvecs : array-like shape (N * 3,)
        The bvecs
    nb_shells: int
        Number of shells.
    nb_points_per_shell: list of ints, len(Ks) = S.
        Number of points per shell.
    weights : array-like, shape (S, S)
        Weighting parameter, control coupling between shells and how this
        balances.

    Returns
    -------
    electrostatic_repulsion: float
        Sum of all interactions between any two vectors.
    """
    nb_points_total = np.sum(nb_points_per_shell)
    indices = np.cumsum(nb_points_per_shell).tolist()
    indices.insert(0, 0)
    weight_matrix = np.zeros((nb_points_total, nb_points_total))
    for s1 in range(nb_shells):
        for s2 in range(nb_shells):
            weight_matrix[indices[s1]:indices[s1 + 1],
                          indices[s2]:indices[s2 + 1]] = weights[s1, s2]
    return _electrostatic_repulsion_energy(bvecs, weight_matrix)


def _electrostatic_repulsion_energy(bvecs, weight_matrix, alpha=1.0):
    """
    Electrostatic-repulsion objective function. The alpha parameter controls
    the power repulsion (energy varies as $1 / ralpha$).

    Parameters
    ---------
    bvecs : array-like shape (N * 3,)
        Vectors, flattened.
    weight_matrix: array-like, shape (N, N)
        The contribution weight of each pair of points.
    alpha : float
        Controls the power of the repulsion. Default is 1.0

    Returns
    -------
    energy : float
        sum of all interactions between any two vectors.
    """
    epsilon = 1e-9
    nb_bvecs = bvecs.shape[0] // 3
    bvecs = bvecs.reshape((nb_bvecs, 3))
    energy = 0.0
    for i in range(nb_bvecs):
        indices = (np.arange(nb_bvecs) > i)
        diffs = ((bvecs[indices] - bvecs[i]) ** 2).sum(1) ** alpha
        sums = ((bvecs[indices] + bvecs[i]) ** 2).sum(1) ** alpha
        energy += (weight_matrix[i, indices] *
                   (1.0 / (diffs + epsilon) + 1.0 / (sums + epsilon))).sum()
    return energy


def _constraint_is_bvec_on_sphere(bvecs, *args):
    """
    Spherical equality constraint. Returns 0 if bvecs lies on the unit sphere.

    This is used as f_eqcons(x, *args), where
    args = (nb_shells, nb_points_per_shell, weights)

    (We do not need args here, but it is sent by scipy and must be kept here.)

    Parameters
    ----------
    bvecs : array-like shape (N * 3)
        Vectors, flattened.

    Returns
    -------
    array shape (N,) : Difference between squared vector norms and 1.
    """
    nb_bvecs = int(bvecs.shape[0] / 3)
    bvecs = bvecs.reshape((nb_bvecs, 3))
    return (bvecs ** 2).sum(1) - 1.0


def _grad_multiple_shell_energy(bvecs, nb_shells, nb_points_per_shell,
                                weights):
    """
    Gradient of the objective function for multiple shells sampling.

    This is called as fprime(x, *args) during optimization, with
    args = (nb_shells, nb_points_per_shell, weights)

    Parameters
    ----------
    bvecs : array-like shape (N * 3,)
        The b-vectors, flattened.
    nb_shells : int
        Number of shells
    nb_points_per_shell : list of ints
        Number of points per shell.
    weights : array-like, shape (S, S)
        Weighting parameter, control coupling between shells and how this
        balances.

    Returns
    -------
    grad_electrostatic_repulsion: float
        Gradient of the objective function.
    """
    nb_bvecs = int(bvecs.shape[0] / 3)
    indices = np.cumsum(nb_points_per_shell).tolist()
    indices.insert(0, 0)
    weight_matrix = np.zeros((nb_bvecs, nb_bvecs))
    for s1 in range(nb_shells):
        for s2 in range(nb_shells):
            weight_matrix[indices[s1]:indices[s1 + 1],
                          indices[s2]:indices[s2 + 1]] = weights[s1, s2]

    return _grad_electrostatic_repulsion_energy(bvecs, weight_matrix)


def _grad_electrostatic_repulsion_energy(bvecs, weight_matrix, alpha=1.0):
    """
    1st-order derivative of electrostatic-like repulsion energy.

    Parameters
    ----------
    bvecs : array-like shape (N * 3,)
        Vectors.
    weight_matrix: array-like, shape (N, N)
        The contribution weight of each pair of bvec.
    alpha : float
        Controls the power of the repulsion. Default is 1.0

    Returns
    -------
    grad : numpy.ndarray
        gradient of the objective function
    """
    nb_bvecs = bvecs.shape[0] // 3
    bvecs = bvecs.reshape((nb_bvecs, 3))
    grad = np.zeros((nb_bvecs, 3))
    for i in range(nb_bvecs):
        indices = (np.arange(nb_bvecs) != i)
        diffs = ((bvecs[indices] - bvecs[i]) ** 2).sum(1) ** (alpha + 1)
        sums = ((bvecs[indices] + bvecs[i]) ** 2).sum(1) ** (alpha + 1)
        grad[i] += (- 2 * alpha * weight_matrix[i, indices] *
                    (bvecs[i] - bvecs[indices]).T / diffs).sum(1)
        grad[i] += (- 2 * alpha * weight_matrix[i, indices] *
                    (bvecs[i] + bvecs[indices]).T / sums).sum(1)
    grad = grad.reshape(nb_bvecs * 3)
    return grad
