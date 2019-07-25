#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum

from dipy.core.sphere import HemiSphere
from dipy.data import get_sphere
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
from dipy.direction.peaks import PeaksAndMetrics
from scilpy.reconst.utils import (find_order_from_nb_coeff,
                                  get_b_matrix, get_maximas)
import nibabel as nib
import numpy as np

class AlgoType(Enum):
    DETERMINISTIC = 'det'
    PROBABILISTIC = 'prob'
    EUDX = 'eudx'


def get_direction_getter(args, mask_data):
    sh_data = nib.load(args.sh_file).get_fdata()
    sphere = HemiSphere.from_sphere(get_sphere(args.sphere))
    theta = get_theta(args.theta, 0, args.step_size, args.algo)

    if args.algo in [AlgoType.DETERMINISTIC, AlgoType.PROBABILISTIC]:
        if args.algo == AlgoType.DETERMINISTIC:
            dg_class = DeterministicMaximumDirectionGetter
        else:
            dg_class = ProbabilisticDirectionGetter
        return dg_class.from_shcoeff(
            shcoeff=sh_data, max_angle=theta, sphere=sphere,
            basis_type=args.sh_basis,
            relative_peak_threshold=args.sf_threshold)

    # Code for type EUDX. We don't use peaks_from_model
    # because we want the peaks from the provided sh.
    sh_shape_3d = sh_data.shape[:-1]
    npeaks = 5
    peak_dirs = np.zeros((sh_shape_3d + (npeaks, 3)))
    peak_values = np.zeros((sh_shape_3d + (npeaks, )))
    peak_indices = np.full((sh_shape_3d + (npeaks, )), -1, dtype='int')
    b_matrix = get_b_matrix(
        find_order_from_nb_coeff(sh_data), sphere, args.sh_basis)

    for idx in np.ndindex(sh_shape_3d):
        if not mask_data[idx]:
            continue

        directions, values, indices = get_maximas(
            sh_data[idx], sphere, b_matrix, args.sf_threshold, 0)
        if values.shape[0] != 0:
            n = min(npeaks, values.shape[0])
            peak_dirs[idx][:n] = directions[:n]
            peak_values[idx][:n] = values[:n]
            peak_indices[idx][:n] = indices[:n]

    dg = PeaksAndMetrics()
    dg.sphere = sphere
    dg.peak_dirs = peak_dirs
    dg.peak_values = peak_values
    dg.peak_indices = peak_indices
    dg.ang_thr = theta
    dg.qa_thr = args.sf_threshold
    return dg


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
    elif tracking_type == AlgoType.PROBABILISTIC:
        theta = 20
    elif tracking_type == AlgoType.EUDX:
        theta = 60
    else:
        theta = 45
    return theta
