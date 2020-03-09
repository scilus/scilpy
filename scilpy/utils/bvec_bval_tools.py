# -*- coding: utf-8 -*-

import logging

import numpy as np

from scilpy.samplingscheme.save_scheme import (save_scheme_bvecs_bvals,
                                               save_scheme_mrtrix)
from scilpy.utils.filenames import split_name_with_nii

DEFAULT_B0_THRESHOLD = 20


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


def check_b0_threshold(force_b0_threshold, bvals_min):
    """Check if the minimal bvalue is under zero or over the default threshold.
    If `force_b0_threshold` is true, don't raise an error even if the minimum
    bvalue is suspiciously high.

    Parameters
    ----------
    force_b0_threshold : bool
        If True, don't raise an error.
    bvals_min : float
        Minimum bvalue.

    Raises
    ------
    ValueError
        If the minimal bvalue is under zero or over the default threshold, and
        `force_b0_threshold` is False.
    """
    if bvals_min != 0:
        if bvals_min < 0 or bvals_min > DEFAULT_B0_THRESHOLD:
            if force_b0_threshold:
                logging.warning(
                    'Warning: Your minimal bval is {}. This is highly '
                    'suspicious. The script will nonetheless proceed since '
                    '--force_b0_threshold was specified.'.format(bvals_min))
            else:
                raise ValueError('The minimal bval is lesser than 0 or '
                                 'greater than {}. This is highly ' +
                                 'suspicious.\n'
                                 'Please check your data to ensure everything '
                                 'is correct.\n'
                                 'Value found: {}\n'
                                 'Use --force_b0_threshold to execute '
                                 'regardless.'
                                 .format(DEFAULT_B0_THRESHOLD, bvals_min))
        else:
            logging.warning('Warning: No b=0 image. Setting b0_threshold to '
                            'the minimum bval: {}'.format(bvals_min))


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
    mrtrix_filename : str, optional
        path to output mrtrix encoding.b file. Default is
        fsl_bvec_filename.b.

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

    basefilename, ext = split_name_with_nii(mrtrix_filename)

    save_scheme_mrtrix(points,
                       shell_idx,
                       bvals,
                       basefilename,
                       verbose=1)


def mrtrix2fsl(mrtrix_filename, fsl_bval_filename=None,
               fsl_bvec_filename=None):
    """
    Convert a mrtrix encoding.b file to fsl dir_grad.bvec/.bval files.

    Parameters
    ----------
    mrtrix_filename : str
        path to mrtrix encoding.b file.
    fsl_bval_filename: str, optional
        path to the output fsl bval file. Default is
        mrtrix_filename.bval.
    fsl_bvec_filename: str, optional
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

    if fsl_bval_filename is None:
        fsl_bval_filename = mrtrix_filename + str(".bval")

    if fsl_bvec_filename is None:
        fsl_bvec_filename = mrtrix_filename + str(".bvec")

    save_scheme_bvecs_bvals(points,
                            shell_idx,
                            bvals,
                            filename_bval=fsl_bval_filename,
                            filename_bvec=fsl_bvec_filename,
                            verbose=1)


# compute the centroid of the bvals given a certain tolerance threshold
def _guess_bvals_centroids(bvals, threshold):
    if not len(bvals):
        raise ValueError('Empty b-values.')

    bval_centroids = [bvals[0]]

    for bval in bvals[1:]:
        diffs = np.abs(np.asarray(bval_centroids) - bval)
        # Found no bval in bval centroids close enough to the current one.
        if not len(np.where(diffs < threshold)[0]):
            bval_centroids.append(bval)

    return np.array(bval_centroids)


# function to estimate the number of shells in the gradient scheme given
# a certain tolerance threshold
def identify_shells(bvals, threshold=40.0):
    centroids = _guess_bvals_centroids(bvals, threshold)

    bvals_for_diffs = np.tile(bvals.reshape(bvals.shape[0], 1),
                              (1, centroids.shape[0]))

    shell_indices = np.argmin(np.abs(bvals_for_diffs - centroids), axis=1)

    return centroids, shell_indices
