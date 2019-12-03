# -*- coding: utf-8 -*-

import logging

import numpy as np

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
    return np.all(np.logical_or(np.abs(bvecs_norm - 1) < 1e-3, bvecs_norm == 0))


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


def check_b0_threshold(args, bvals_min):
    if bvals_min != 0:
        if bvals_min < 0 or bvals_min > DEFAULT_B0_THRESHOLD:
            if args.force_b0_threshold:
                logging.warning(
                    'Warning: Your minimal bvalue is {}. This is highly '
                    'suspicious. The script will nonetheless proceed since '
                    '--force_b0_threshold was specified.'.format(bvals_min))
            else:
                raise ValueError('The minimal bvalue is lesser than 0 or '
                                 'greater than {}. This is highly suspicious.\n'
                                 'Please check your data to ensure everything '
                                 'is correct.\n'
                                 'Value found: {}\n'
                                 'Use --force_b0_threshold to run the script '
                                 'regardless.'
                                 .format(DEFAULT_B0_THRESHOLD, bvals_min))
        else:
            logging.warning('Warning: No b=0 image. Setting b0_threshold to '
                            'the minimum bvalue: {}'.format(bvals_min))


def get_shell_indices(bvals, shell, tol=10):
    return np.where(
        np.logical_and(bvals < shell + tol, bvals > shell - tol))[0]


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
