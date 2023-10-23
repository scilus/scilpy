# -*- coding: utf-8 -*-

import numpy as np


def random_uniform_on_sphere(nb_vectors):
    """
    Creates a set of K pseudo-random unit vectors, following a uniform
    distribution on the sphere.

    Parameters
    ----------
    nb_vectors: int
        Number of vectors

    Returns
    -------
    bvecs: nd.array
        pseudo-random unit vector
    """
    phi = 2 * np.pi * np.random.rand(nb_vectors)

    r = 2 * np.sqrt(np.random.rand(nb_vectors))
    theta = 2 * np.arcsin(r / 2)

    bvecs = np.zeros((nb_vectors, 3))
    bvecs[:, 0] = np.sin(theta) * np.cos(phi)
    bvecs[:, 1] = np.sin(theta) * np.sin(phi)
    bvecs[:, 2] = np.cos(theta)

    return bvecs
