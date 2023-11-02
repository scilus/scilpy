# -*- coding: utf-8 -*-

import logging
from enum import Enum

import numpy as np

from scilpy.io.gradient_table import (save_gradient_sampling_fsl,
                                      save_gradient_sampling_mrtrix)

DEFAULT_B0_THRESHOLD = 20


class B0ExtractionStrategy(Enum):
    FIRST = "first"
    MEAN = "mean"
    ALL = "all"


def is_normalized_bvecs(bvecs):
    """
    Check if b-vectors are normalized.

    Parameters
    ----------
    bvecs : (N, 3) array
        input b-vectors (N, 3) array

    Returns
    -------
    True/False
    """

    bvecs_norm = np.linalg.norm(bvecs, axis=1)
    return np.all(np.logical_or(np.abs(bvecs_norm - 1) < 1e-3,
                                bvecs_norm == 0))


def normalize_bvecs(bvecs, filename=None):
    """
    Normalize b-vectors

    Parameters
    ----------
    bvecs : (N, 3) array
        input b-vectors (N, 3) array
    filename : string
        output filename where to save the normalized bvecs

    Returns
    -------
    bvecs : (N, 3)
       normalized b-vectors
    """

    bvecs_norm = np.linalg.norm(bvecs, axis=1)
    idx = bvecs_norm != 0
    bvecs[idx] /= bvecs_norm[idx, None]

    if filename is not None:
        logging.info('Saving new bvecs: {}'.format(filename))
        np.savetxt(filename, np.transpose(bvecs), "%.8f")

    return bvecs


def check_b0_threshold(
    force_b0_threshold, bvals_min, b0_thr=DEFAULT_B0_THRESHOLD
):
    """Check if the minimal bvalue is under zero or over the threshold.
    If `force_b0_threshold` is true, don't raise an error even if the minimum
    bvalue is over the threshold.

    Parameters
    ----------
    force_b0_threshold : bool
        If True, don't raise an error.
    bvals_min : float
        Minimum bvalue.
    b0_thr : float
        Maximum bvalue considered as a b0.

    Raises
    ------
    ValueError
        If the minimal bvalue is over the threshold, and
        `force_b0_threshold` is False.
    """
    if b0_thr > DEFAULT_B0_THRESHOLD:
        logging.warning(
            'Warning: Your defined threshold is {}. This is suspicious. We '
            'recommend using volumes with bvalues no higher '
            'than {} as b0s.'.format(b0_thr, DEFAULT_B0_THRESHOLD)
        )

    if bvals_min < 0:
        logging.warning(
            'Warning: Your dataset contains negative b-values (minimal '
            'bvalue of {}). This is suspicious. We recommend you check '
            'your data.')

    if bvals_min > b0_thr:
        if force_b0_threshold:
            logging.warning(
                'Warning: Your minimal bvalue is {}, but the threshold '
                'is set to {}. Since --force_b0_threshold was specified, '
                'the script will proceed with a threshold of {}'
                '.'.format(bvals_min, b0_thr, bvals_min))
            return bvals_min
        else:
            raise ValueError('The minimal bvalue ({}) is greater than the '
                             'threshold ({}). No b0 volumes can be found.\n'
                             'Please check your data to ensure everything '
                             'is correct.\n'
                             'Use --force_b0_threshold to execute '
                             'regardless.'
                             .format(bvals_min, b0_thr))

    return b0_thr


def get_shell_indices(bvals, shell, tol=10):
    """
    Get shell indices

    Parameters
    ----------
    bvals: array (N,)
        array of bvals
    shell: list
        list of bvals
    tol: int
        tolerance to accept a bval

    Returns
    -------
        numpy.ndarray where shells are found
    """

    return np.where(
        np.logical_and(bvals < shell + tol, bvals > shell - tol))[0]


def fsl2mrtrix(fsl_bval_filename, fsl_bvec_filename, mrtrix_filename):
    """
    Convert a fsl dir_grad.bvec/.bval files to mrtrix encoding.b file.

    Parameters
    ----------
    fsl_bval_filename: str
        path to input fsl bval file.
    fsl_bvec_filename: str
        path to input fsl bvec file.
    mrtrix_filename : str
        path to output mrtrix encoding.b file.

    Returns
    -------
    """

    shells = np.loadtxt(fsl_bval_filename)
    points = np.loadtxt(fsl_bvec_filename)
    bvals = np.unique(shells).tolist()

    if not points.shape[0] == 3:
        points = points.transpose()
        logging.warning('WARNING: Your bvecs seem transposed. ' +
                        'Transposing them.')

    shell_idx = [int(np.where(bval == bvals)[0]) for bval in shells]
    save_gradient_sampling_mrtrix(points,
                                  shell_idx,
                                  bvals,
                                  mrtrix_filename)


def mrtrix2fsl(mrtrix_filename, fsl_bval_filename=None,
               fsl_bvec_filename=None):
    """
    Convert a mrtrix encoding.b file to fsl dir_grad.bvec/.bval files.

    Parameters
    ----------
    mrtrix_filename : str
        path to mrtrix encoding.b file.
    fsl_bval_filename: str
        path to the output fsl bval file. Default is
        mrtrix_filename.bval.
    fsl_bvec_filename: str
        path to the output fsl bvec file. Default is
        mrtrix_filename.bvec.
    Returns
    -------
    """

    mrtrix_b = np.loadtxt(mrtrix_filename)
    if not len(mrtrix_b.shape) == 2 or not mrtrix_b.shape[1] == 4:
        raise ValueError('mrtrix file must have 4 columns')

    points = np.array([mrtrix_b[:, 0], mrtrix_b[:, 1], mrtrix_b[:, 2]])
    shells = np.array(mrtrix_b[:, 3])

    bvals = np.unique(shells).tolist()
    shell_idx = [int(np.where(bval == bvals)[0]) for bval in shells]

    save_gradient_sampling_fsl(points,
                               shell_idx,
                               bvals,
                               filename_bval=fsl_bval_filename,
                               filename_bvec=fsl_bvec_filename)


def identify_shells(bvals, threshold=40.0, roundCentroids=False, sort=False):
    """
    Guessing the shells from the b-values. Returns the list of shells and, for
    each b-value, the associated shell.

    Starting from the first shell as holding the first b-value in bvals,
    the next b-value is considered on the same shell if it is closer than
    threshold, or else we consider that it is on another shell. This is an
    alternative to K-means considering we don't already know the number of
    shells K.

    Note. This function should be added in Dipy soon.

    Parameters
    ----------
    bvals: array (N,)
        Array of bvals
    threshold: float
        Limit value to consider that a b-value is on an existing shell. Above
        this limit, the b-value is placed on a new shell.
    roundCentroids: bool
        If true will round shell values to the nearest 10.
    sort: bool
        Sort centroids and shell_indices associated.

    Returns
    -------
    centroids: array (K)
        Array of centroids. Each centroid is a b-value representing the shell.
        K is the number of identified shells.
    shell_indices: array (N,)
        For each bval, the associated centroid K.
    """
    if len(bvals) == 0:
        raise ValueError('Empty b-values.')

    # Finding centroids
    bval_centroids = [bvals[0]]
    for bval in bvals[1:]:
        diffs = np.abs(np.asarray(bval_centroids, dtype=float) - bval)
        if not len(np.where(diffs < threshold)[0]):
            # Found no bval in bval centroids close enough to the current one.
            # Create new centroid (i.e. new shell)
            bval_centroids.append(bval)
    centroids = np.array(bval_centroids)

    # Identifying shells
    bvals_for_diffs = np.tile(bvals.reshape(bvals.shape[0], 1),
                              (1, centroids.shape[0]))

    shell_indices = np.argmin(np.abs(bvals_for_diffs - centroids), axis=1)

    if roundCentroids:
        centroids = np.round(centroids, decimals=-1)

    if sort:
        sort_index = np.argsort(centroids)
        sorted_centroids = np.zeros(centroids.shape)
        sorted_indices = np.zeros(shell_indices.shape)
        for i in range(len(centroids)):
            sorted_centroids[i] = centroids[sort_index[i]]
            sorted_indices[shell_indices == i] = sort_index[i]
        return sorted_centroids, sorted_indices

    return centroids, shell_indices


def flip_gradient_sampling(bvecs, axes, sampling_type):
    """
    Flip bvecs on chosen axis.

    Parameters
    ----------
    bvecs: np.ndarray
        Loaded bvecs. In the case 'mrtrix' the bvecs actually also contain the
        bvals.
    axes: list of int
        List of axes to flip (e.g. [0, 1])
    sampling_type: str
        Either 'mrtrix' or 'fsl'.
    """
    assert sampling_type in ['mrtrix', 'fsl']
    if sampling_type == 'mrtrix':
        for axis in axes:
            bvecs[:, axis] *= -1
    else:
        for axis in axes:
            bvecs[axis, :] *= -1
    return bvecs


def swap_gradient_axis(bvecs, final_order, sampling_type):
    """
    Swap bvecs.

    Parameters
    ----------
    bvecs: np.array
        Loaded bvecs. In the case 'mrtrix' the bvecs actually also contain the
        bvals.
    final_order: new order
        Final order (ex, 2 1 0)
    sampling_type: str
        Either 'mrtrix' or 'fsl'.
    """
    new_bvecs = np.copy(bvecs)
    assert sampling_type in ['mrtrix', 'fsl']
    if sampling_type == 'mrtrix':
        new_bvecs[:, 0] = bvecs[:, final_order[0]]
        new_bvecs[:, 1] = bvecs[:, final_order[1]]
        new_bvecs[:, 2] = bvecs[:, final_order[2]]
    else:
        new_bvecs[0, :] = bvecs[final_order[0], :]
        new_bvecs[1, :] = bvecs[final_order[1], :]
        new_bvecs[2, :] = bvecs[final_order[2], :]
    return new_bvecs


def extract_bvals(bvals, tolerance, bvals_to_extract):
    """
    Return bvals equal to a list of chosen bvals, up to a tolerance.

    Parameters
    ----------
    bvals: np.array
        All the b-values.
    tolerance: float
        The tolerance
    bvals_to_extract: list
        The shells of interest.
    """
    # Find the volume indices that correspond to the shells to extract.
    sorted_centroids, sorted_indices = identify_shells(bvals, tolerance,
                                                       sort=True)

    bvals_to_extract = np.sort(bvals_to_extract)
    nb_new_shells = np.shape(bvals_to_extract)[0]

    logging.info("number of shells: {}".format(nb_new_shells))
    logging.info("bvals to extract: {}".format(bvals_to_extract))
    logging.info("estimated centroids: {}".format(sorted_centroids))
    logging.info("original bvals: {}".format(bvals))
    logging.info("selected indices: {}".format(sorted_indices))

    new_bvals = bvals
    for i in range(nb_new_shells):
        if np.abs(sorted_centroids[i] - bvals_to_extract[i]) <= tolerance:
            new_bvals[np.where(sorted_indices == i)] = bvals_to_extract[i]
        else:
            raise ValueError("No bvals found on shell #{}: {}: "
                             "tolerance is too low?"
                             .format(i, bvals[i]))
    return new_bvals
