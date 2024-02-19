# -*- coding: utf-8 -*-

import numpy as np


def random_uniform_on_sphere(nb_vectors):
    """
    Creates a set of K pseudo-random unit vectors, following a uniform
    distribution on the sphere. Reference: Emmanuel Caruyer's code
    (https://github.com/ecaruyer).

    This is not intended to create a perfect result. It's usually the
    initialization step of a repulsion strategy.

    Parameters
    ----------
    nb_vectors: int
        Number of vectors

    Returns
    -------
    bvecs: nd.array of shape (nb_vectors, 3)
        Pseudo-random unit vectors
    """
    # Note. Caruyer's docstring says it's a uniform on the half-sphere, but
    # plotted a few results: it is one the whole sphere.
    phi = 2 * np.pi * np.random.rand(nb_vectors)

    r = 2 * np.sqrt(np.random.rand(nb_vectors))
    theta = 2 * np.arcsin(r / 2)

    # See here:
    # https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    # They seem to be using something like this instead:
    # theta = np.arccos(2 * np.random.rand(nb_vectors) - 1.0)

    bvecs = np.zeros((nb_vectors, 3))
    bvecs[:, 0] = np.sin(theta) * np.cos(phi)
    bvecs[:, 1] = np.sin(theta) * np.sin(phi)
    bvecs[:, 2] = np.cos(theta)

    return bvecs


def get_new_order_philips(philips_table, dwi, bvals, bvecs):
    """
    Find the sorting order that could be applied to the bval and bvec files to
    obtain the same order as in the philips table.

    Parameters
    ----------
    philips_table: nd.array
        Philips gradient table, of shape (N, 4), coming from a Philips machine
        with SoftwareVersions < 5.6.
    dwi: nibabel image
        dwi of shape (x, y, z, N). Only used to confirm the dwi's shape.
    bvals : array, (N,)
        bvals
    bvecs : array, (N, 3)
        bvecs

    Returns
    -------
    new_index: nd.array
        New index to reorder bvals/bvec
    """
    # Check number of gradients, bvecs, bvals, dwi and oTable
    if not (len(bvecs) == dwi.shape[3] == len(bvals) == len(philips_table)):
        raise ValueError('bvec/bval/dwi and original table do not contain '
                         'the same number of gradients.')

    # Check bvals
    philips_bval = np.unique(philips_table[:, 3])

    philips_dwi_shells = philips_bval[philips_bval > 1]
    philips_b0s = philips_bval[philips_bval < 1]

    dwi_shells = np.unique(bvals[bvals > 1])
    b0s = np.unique(bvals[bvals < 1])

    if len(philips_dwi_shells) != len(dwi_shells) or \
       len(philips_b0s) != len(b0s):
        raise ValueError('bvec/bval/dwi and original table do not contain '
                         'the same shells.')

    new_index = np.zeros(bvals.shape)

    for nbval in philips_bval:
        curr_bval = np.where(bvals == nbval)[0]
        curr_bval_table = np.where(philips_table[:, 3] == nbval)[0]

        if len(curr_bval) != len(curr_bval_table):
            raise ValueError('bval/bvec and orginal table do not contain '
                             'the same number of gradients for shell {}.'
                             .format(curr_bval))

        new_index[curr_bval_table] = curr_bval

    return new_index.astype(int)
