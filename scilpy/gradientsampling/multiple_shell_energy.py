# -*- coding: utf-8 -*-

import numpy as np


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
    $\\partial f_i / \\partial x_j$
    """
    N = bvecs.shape[0] / 3
    bvecs = bvecs.reshape((N, 3))
    bvecs = (bvecs.T / np.sqrt((bvecs ** 2).sum(1))).T
    grad = np.zeros((N, N * 3))
    for i in range(3):
        grad[:, i * N:(i+1) * N] = np.diag(bvecs[:, i])
    return grad


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

