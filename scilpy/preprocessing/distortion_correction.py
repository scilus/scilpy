# -*- coding: utf-8 -*-

from builtins import range

import numpy as np


def create_acqparams(gtab, readout, encoding_direction, keep_all_b0s=True):
    nb_b0s = 1
    if keep_all_b0s:
        nb_b0s = len(np.where(gtab.b0s_mask)[0])
    acqparams = np.zeros((nb_b0s + 1, 4))
    acqparams[:, 3] = readout

    enum_direction = {'x': 0, 'y': 1, 'z': 2}
    acqparams[0:-1, enum_direction[encoding_direction]] = 1
    acqparams[-1, enum_direction[encoding_direction]] = -1

    return acqparams


def create_index(bvals):
    index = np.ones(len(bvals), dtype=np.int).tolist()

    return index


def create_non_zero_norm_bvecs(bvecs):
    # Set the bvecs to an epsilon if the norm is 0.
    # Mandatory to compute topup/eddy
    for i in range(len(bvecs)):
        if np.linalg.norm(bvecs[i, :]) < 0.00000001:
            bvecs[i, :] += 0.00000001

    return bvecs
