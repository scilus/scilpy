# -*- coding: utf-8 -*-

################################################
# Author: Emmanuel Caruyer <caruyer@gmail.com> #
#                                              #
# Code to generate multiple-shell sampling     #
# schemes, with optimal angular coverage. This #
# implements the method described in Caruyer   #
# et al., MRM 69(6), pp. 1534-1540, 2013.      #
# This software comes with no warranty, etc.   #
################################################
from scipy import optimize
import numpy as np


def equality_constraints(bvecs, *args):
    """
    Spherical equality constraint. Returns 0 if bvecs lies on the unit sphere.

    Parameters
    ----------
    bvecs : array-like shape (N * 3)

    Returns
    -------
    array shape (N,) : Difference between squared vector norms and 1.
    """
    N = int(bvecs.shape[0] / 3)
    bvecs = bvecs.reshape((N, 3))
    return (bvecs ** 2).sum(1) - 1.0


def grad_equality_constraints(bvecs, *args):
    """
    Return normals to the surface constraint (wich corresponds to
    the gradient of the implicit function).

    Parameters
    ----------
    bvecs : array-like shape (N * 3)

    Returns
    -------
    array shape (N, N * 3). grad[i, j] contains
    $\partial f_i / \partial x_j$
    """
    N = bvecs.shape[0] / 3
    bvecs = bvecs.reshape((N, 3))
    bvecs = (bvecs.T / np.sqrt((bvecs ** 2).sum(1))).T
    grad = np.zeros((N, N * 3))
    for i in range(3):
        grad[:, i * N:(i+1) * N] = np.diag(bvecs[:, i])
    return grad


def electrostatic_repulsion(bvecs, weight_matrix, alpha=1.0):
    """
    Electrostatic-repulsion objective function. The alpha parameter controls
    the power repulsion (energy varies as $1 / ralpha$).

    Parameters
    ---------
    bvecs : array-like shape (N * 3,)
        Vectors.
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
    N = bvecs.shape[0] // 3
    bvecs = bvecs.reshape((N, 3))
    energy = 0.0
    for i in range(N):
        indices = (np.arange(N) > i)
        diffs = ((bvecs[indices] - bvecs[i]) ** 2).sum(1) ** alpha
        sums = ((bvecs[indices] + bvecs[i]) ** 2).sum(1) ** alpha
        energy += (weight_matrix[i, indices] *
                   (1.0 / (diffs + epsilon) + 1.0 / (sums + epsilon))).sum()
    return energy


def grad_electrostatic_repulsion(bvecs, weight_matrix, alpha=1.0):
    """
    1st-order derivative of electrostatic-like repulsion energy.

    Parameters
    ----------
    bvecs : array-like shape (N * 3,)
        Vectors.
    weight_matrix: array-like, shape (N, N)
        The contribution weight of each pair of points.
    alpha : floating-point. controls the power of the repulsion. Default is 1.0

    Returns
    -------
    grad : numpy.ndarray
        gradient of the objective function
    """
    N = bvecs.shape[0] // 3
    bvecs = bvecs.reshape((N, 3))
    grad = np.zeros((N, 3))
    for i in range(N):
        indices = (np.arange(N) != i)
        diffs = ((bvecs[indices] - bvecs[i]) ** 2).sum(1) ** (alpha + 1)
        sums = ((bvecs[indices] + bvecs[i]) ** 2).sum(1) ** (alpha + 1)
        grad[i] += (- 2 * alpha * weight_matrix[i, indices] *
                    (bvecs[i] - bvecs[indices]).T / diffs).sum(1)
        grad[i] += (- 2 * alpha * weight_matrix[i, indices] *
                    (bvecs[i] + bvecs[indices]).T / sums).sum(1)
    grad = grad.reshape(N * 3)
    return grad


def cost(bvecs, S, Ks, weights):
    """
    Objective function for multiple-shell energy.

    Parameters
    ----------
    bvecs : array-like shape (N * 3,)

    S: int
        Number of shells.
    Ks: list of ints, len(Ks) = S. Number of points per shell.
    weights : array-like, shape (S, S)
        Weighting parameter, control coupling between shells and how this
        balances.

    Returns
    -------
    electrostatic_repulsion: float
        sum of all interactions between any two vectors.
    """
    K = np.sum(Ks)
    indices = np.cumsum(Ks).tolist()
    indices.insert(0, 0)
    weight_matrix = np.zeros((K, K))
    for s1 in range(S):
        for s2 in range(S):
            weight_matrix[indices[s1]:indices[s1 + 1],
                          indices[s2]:indices[s2 + 1]] = weights[s1, s2]
    return electrostatic_repulsion(bvecs, weight_matrix)


def grad_cost(bvecs, S, Ks, weights):
    """
    gradient of the objective function for multiple shells sampling.

    Parameters
    ----------
    bvecs : array-like shape (N * 3,)
    S : int
        number of shells
    Ks : list of ints
        len(Ks) = S. Number of points per shell.
    weights : array-like, shape (S, S)
        weighting parameter, control coupling between shells and how this
        balances.

    Returns
    -------
    grad_electrostatic_repulsion: float
        gradient of the objective function
    """
    K = int(bvecs.shape[0] / 3)
    indices = np.cumsum(Ks).tolist()
    indices.insert(0, 0)
    weight_matrix = np.zeros((K, K))
    for s1 in range(S):
        for s2 in range(S):
            weight_matrix[indices[s1]:indices[s1 + 1],
                          indices[s2]:indices[s2 + 1]] = weights[s1, s2]

    return grad_electrostatic_repulsion(bvecs, weight_matrix)


def multiple_shell(nb_shells, nb_points_per_shell, weights, max_iter=100,
                   verbose=2):
    """
    Creates a set of sampling directions on the desired number of shells.

    Parameters
    ----------
    nb_shells : the number of shells
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
    K = np.sum(nb_points_per_shell)

    # Initialized with random directions
    bvecs = random_uniform_on_sphere(K)
    bvecs = bvecs.reshape(K * 3)

    bvecs = optimize.fmin_slsqp(cost, bvecs.reshape(K * 3),
                                f_eqcons=equality_constraints,
                                fprime=grad_cost,
                                iter=max_iter,
                                acc=1.0e-9,
                                args=(nb_shells,
                                      nb_points_per_shell,
                                      weights),
                                iprint=verbose)
    bvecs = bvecs.reshape((K, 3))
    bvecs = (bvecs.T / np.sqrt((bvecs ** 2).sum(1))).T
    return bvecs


def write_multiple_shells(bvecs, nb_shells, nb_points_per_shell, filename):
    """
    Export multiple shells to text file.

    Parameters
    ----------
    bvecs : array-like shape (K, 3)
        vectors
    nb_shells: int
        Number of shells
    nb_points_per_shell: array-like shape (nb_shells, )
        A list of integers containing the number of points on each shell.
    filename : str
        output filename
    """
    datafile = open(filename, 'w')
    datafile.write('#shell-id\tx\ty\tz\n')
    k = 0
    for s in range(nb_shells):
        for n in range(nb_points_per_shell[s]):
            datafile.write("%d\t%f\t%f\t%f\n" %
                           (s, bvecs[k, 0], bvecs[k, 1], bvecs[k, 2]))
            k += 1
    datafile.close()


def random_uniform_on_sphere(K):
    """
    Creates a set of K pseudo-random unit vectors, following a uniform
    distribution on the sphere.

    Parameters
    ----------
    K: int
        Number of vectors

    Returns
    -------
    bvecs: nd.array
        pseudo-random unit vector
    """
    phi = 2 * np.pi * np.random.rand(K)

    r = 2 * np.sqrt(np.random.rand(K))
    theta = 2 * np.arcsin(r / 2)

    bvecs = np.zeros((K, 3))
    bvecs[:, 0] = np.sin(theta) * np.cos(phi)
    bvecs[:, 1] = np.sin(theta) * np.sin(phi)
    bvecs[:, 2] = np.cos(theta)

    return bvecs


def compute_weights(nb_shells, nb_points_per_shell, shell_groups, alphas):
    """
    Computes the weights array from a set of shell groups to couple, and
    coupling weights.

    Parameters
    ----------
    nb_shells: int
        Number of shells
    nb_points_per_shell: int
        Number of points per shell
    shell_groups: list
        list of group of shells
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
