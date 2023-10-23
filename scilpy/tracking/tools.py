# -*- coding: utf-8 -*-

import numpy as np


def get_theta(requested_theta, tracking_type):
    if requested_theta is not None:
        theta = requested_theta
    elif tracking_type == 'prob':
        theta = 20
    elif tracking_type == 'eudx':
        theta = 60
    else:
        theta = 45
    return theta


def sample_distribution(dist):
    """
    Parameters
    ----------
    dist: numpy.array
        The empirical distribution to sample from.

    Return
    ------
    ind: int
        The index of the sampled element.
    """
    cdf = dist.cumsum()
    if cdf[-1] == 0:
        return None
    return cdf.searchsorted(np.random.random() * cdf[-1])
