#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

DETERMINISTIC = 'det'
PROBABILISTIC = 'prob'
EUDX = 'eudx'


def get_max_angle_from_curvature(curvature, step_size):
    """
    Parameters
    ----------
    curvature: float
        Minimum radius of curvature in mm.
    step_size: float
        The tracking step size in mm.

    Return
    ------
    theta: float
        The maximum deviation angle in radian,
        given the radius curvature and the step size.
    """
    theta = 2. * np.arcsin(step_size / (2. * curvature))
    if np.isnan(theta) or theta > np.pi / 2 or theta <= 0:
        theta = np.pi / 2.0
    return theta


def get_theta(requested_theta, curvature, step_size, tracking_type):
    """
    Parameters
    ----------
    requested_theta : float or None
        Desired angular threshold (or None if unused)
    curvature: float
        Minimum radius of curvature in mm.
    step_size: float
        The tracking step size in mm.
    tracking_type: str
        Choice of algorithm for tractography
            [PROBABILISTIC, DETERMINISTIC, EUDX]

    Return
    ------
    theta: float
        The maximum deviation angle in degree
    """
    if requested_theta is not None:
        theta = requested_theta
    elif curvature > 0:
        theta = np.rad2deg(get_max_angle_from_curvature(curvature, step_size))
    elif tracking_type == PROBABILISTIC:
        theta = 20
    elif tracking_type == EUDX:
        theta = 60
    else:
        theta = 45
    return theta
