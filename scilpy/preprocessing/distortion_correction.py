# -*- coding: utf-8 -*-

import numpy as np


def create_acqparams(readout, encoding_direction, nb_b0s=1, nb_rev_b0s=1):
    """
    Create acqparams for Topup and Eddy

    Parameters
    ----------
    readout: float
        Readout time
    encoding_direction: string
        Encoding direction (x, y or z)
    nb_b0s: int
        Number of B=0 images
    nb_rev_b0s: int
        number of reverse b=0 images
    Returns
    -------
    acqparams: np.array
        acqparams
    """
    acqparams = np.zeros((nb_b0s + nb_rev_b0s, 4))
    acqparams[:, 3] = readout

    enum_direction = {'x': 0, 'y': 1, 'z': 2}
    acqparams[0:nb_b0s, enum_direction[encoding_direction]] = 1
    if nb_rev_b0s > 0:
        acqparams[nb_b0s:, enum_direction[encoding_direction]] = -1

    return acqparams


def create_index(bvals):
    """
    Create index of bvals for Eddy

    Parameters
    ----------
    bvals: np.array
        b-values

    Returns
    -------
    index: np.array
    """
    index = np.ones(len(bvals), dtype=np.int).tolist()

    return index


def create_non_zero_norm_bvecs(bvecs):
    """
    Add an epsilon to bvecs with a non zero norm.
    Mandatory for Topup and Eddy

    Parameters
    ----------
    bvecs: np.array
        b-vectors
    Returns
    -------
    bvecs: np.array
        b-vectors with an epsilon
    """
    # Set the bvecs to an epsilon if the norm is 0.
    # Mandatory to compute topup/eddy
    for i in range(len(bvecs)):
        if np.linalg.norm(bvecs[i, :]) < 0.00000001:
            bvecs[i, :] += 0.00000001

    return bvecs
