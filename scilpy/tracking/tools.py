# -*- coding: utf-8 -*-
import logging

import nibabel as nib
import numpy as np

from dipy.core.sphere import HemiSphere
from dipy.data import get_sphere
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter,
                            PTTDirectionGetter)
from dipy.direction.peaks import PeaksAndMetrics

from scilpy.reconst.utils import (find_order_from_nb_coeff,
                                  get_b_matrix, get_maximas)


def get_direction_getter(
    in_odf, algo, sphere, sub_sphere, theta, sh_basis,
    voxel_size, sf_threshold
):
    """ Return the direction getter object.

    Parameters
    ----------
    in_odf: str
        Path to the input odf file.
    algo: str
        Algorithm to use for tracking. Can be 'det', 'prob', 'ptt' or 'eudx'.
    sphere: str
        Name of the sphere to use for tracking.
    sub_sphere: int
        Number of subdivisions to use for the sphere.
    theta: float
        Angle threshold for tracking.
    sh_basis: str
        Name of the sh basis to use for tracking.
    voxel_size: float
        Voxel size of the input data.
    sf_threshold: float
        Spherical function-amplitude threshold for tracking.

    Return
    ------
    dg: dipy.direction.DirectionGetter
        The direction getter object.
    """

    odf_data = nib.load(in_odf).get_fdata(dtype=np.float32)

    sphere = HemiSphere.from_sphere(
        get_sphere(sphere)).subdivide(sub_sphere)

    # Theta depends on user choice and algorithm
    theta = get_theta(theta, algo)

    # Heuristic to find out if the input are peaks or fodf
    non_zeros_count = np.count_nonzero(np.sum(odf_data, axis=-1))
    non_first_val_count = np.count_nonzero(np.argmax(odf_data, axis=-1))

    if algo in ['det', 'prob', 'ptt']:
        if non_first_val_count / non_zeros_count > 0.5:
            logging.warning('Input detected as peaks. Input should be'
                            'fodf for det/prob, verify input just in case.')

        kwargs = {}
        if algo == 'ptt':
            dg_class = PTTDirectionGetter
            # Considering the step size usually used, the probe length
            # can be set as the voxel size.
            kwargs = {'probe_length': voxel_size}
        elif algo == 'det':
            dg_class = DeterministicMaximumDirectionGetter
        else:
            dg_class = ProbabilisticDirectionGetter
        return dg_class.from_shcoeff(
            shcoeff=odf_data, max_angle=theta, sphere=sphere,
            basis_type=sh_basis,
            relative_peak_threshold=sf_threshold, **kwargs)
    elif algo == 'eudx':
        # Code for type EUDX. We don't use peaks_from_model
        # because we want the peaks from the provided sh.
        odf_shape_3d = odf_data.shape[:-1]
        dg = PeaksAndMetrics()
        dg.sphere = sphere
        dg.ang_thr = theta
        dg.qa_thr = sf_threshold

        # Heuristic to find out if the input are peaks or fodf
        # fodf are always around 0.15 and peaks around 0.75
        if non_first_val_count / non_zeros_count > 0.5:
            logging.info('Input detected as peaks.')
            nb_peaks = odf_data.shape[-1] // 3
            slices = np.arange(0, 15+1, 3)
            peak_values = np.zeros(odf_shape_3d+(nb_peaks,))
            peak_indices = np.zeros(odf_shape_3d+(nb_peaks,))

            for idx in np.argwhere(np.sum(odf_data, axis=-1)):
                idx = tuple(idx)
                for i in range(nb_peaks):
                    peak_values[idx][i] = np.linalg.norm(
                        odf_data[idx][slices[i]:slices[i+1]], axis=-1)
                    peak_indices[idx][i] = sphere.find_closest(
                        odf_data[idx][slices[i]:slices[i+1]])

            dg.peak_dirs = odf_data
        else:
            # If the input is not peaks, we assume it is fodf
            # and we compute the peaks from the fodf.
            logging.info('Input detected as fodf.')
            npeaks = 5
            peak_dirs = np.zeros((odf_shape_3d + (npeaks, 3)))
            peak_values = np.zeros((odf_shape_3d + (npeaks, )))
            peak_indices = np.full((odf_shape_3d + (npeaks, )), -1,
                                   dtype='int')
            b_matrix = get_b_matrix(
                find_order_from_nb_coeff(odf_data), sphere, sh_basis)

            for idx in np.argwhere(np.sum(odf_data, axis=-1)):
                idx = tuple(idx)
                directions, values, indices = get_maximas(odf_data[idx],
                                                          sphere, b_matrix,
                                                          sf_threshold, 0)
                if values.shape[0] != 0:
                    n = min(npeaks, values.shape[0])
                    peak_dirs[idx][:n] = directions[:n]
                    peak_values[idx][:n] = values[:n]
                    peak_indices[idx][:n] = indices[:n]

            dg.peak_dirs = peak_dirs

        dg.peak_values = peak_values
        dg.peak_indices = peak_indices

        return dg


def get_theta(requested_theta, tracking_type):
    if requested_theta is not None:
        theta = requested_theta
    elif tracking_type == 'ptt':
        theta = 20
    elif tracking_type == 'prob':
        theta = 20
    elif tracking_type == 'eudx':
        theta = 60
    else:
        theta = 45
    return theta


def sample_distribution(dist, random_generator: np.random.Generator):
    """
    Parameters
    ----------
    dist: numpy.array
        The empirical distribution to sample from.
    random_generator: numpy Generator

    Return
    ------
    ind: int
        The index of the sampled element.
    """
    cdf = dist.cumsum()
    if cdf[-1] == 0:
        return None

    return cdf.searchsorted(random_generator.random() * cdf[-1])
