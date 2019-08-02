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
                                 'greater than {}. This is highly ' +
                                 'suspicious.\n'
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
    fsl_bval = np.loadtxt(fsl_bval_filename)
    fsl_bvec = np.loadtxt(fsl_bvec_filename)

    if not fsl_bvec.shape[0] == 3:
        fsl_bvec = fsl_bvec.transpose()
        logging.warning('WARNING: Your bvecs seem transposed. ' +
                        'Transposing them.')

    if not fsl_bval.shape[0] == 1:
        fsl_bval = fsl_bval.transpose()
        logging.warning('WARNING: Your bvals seem transposed. ' +
                        'Transposing them.')

    if not fsl_bvec[0].shape == fsl_bval.shape:
        raise ValueError('Bvec and Bval files have a different ' +
                         'number of entries.')

    mrtrix_b = np.array([fsl_bvec[0], fsl_bvec[1],
                         fsl_bvec[2], fsl_bval]).transpose()

    if mrtrix_filename is None:
        mrtrix_filename = fsl_bvec_filename + ".b"

    np.savetxt(mrtrix_filename, mrtrix_b, "%.8f %.8f %.8f %f")

    return


def mrtrix2fsl(mrtrix_filename, fsl_bval_filename=None, fsl_bvec_filename=None,
               fsl_base_filename=None):
    """
    Convert a mrtrix encoding.b file to fsl dir_grad.bvec/.bval files.

    Parameters
    ----------
    mrtrix_filename : str
        path to mrtrix encoding.b file.
    fsl_bval_filename: str, optional
        path to output fsl bval file. Default is
        mrtrix_filename.bval.
    fsl_bvec_filename: str, optional
        path to output fsl bvec file. Default is
        mrtrix_filename.bvec.
    fsl_base_filename: str, optional
        path to the output fsl bvec/.bval files. Default is
        mrtrix_filename.bval/.bvec. Used if fsl_bval_filename
        and fsl_bvec_filename are not specified.

    Returns
    -------

    """

    mrtrix_b = np.loadtxt(mrtrix_filename)
    if not len(mrtrix_b.shape) == 2 or not mrtrix_b.shape[1] == 4:
        raise ValueError('mrtrix file must have 4 columns')

    bvec = np.array([mrtrix_b[:, 0], mrtrix_b[:, 1], mrtrix_b[:, 2]])
    bval = np.array([mrtrix_b[:, 3]])

    if fsl_bval_filename is None:
        if fsl_base_filename is None:
            fsl_bval_filename = mrtrix_filename + str(".bval")
        else:
            fsl_bval_filename = fsl_base_filename + str(".bval")
    if fsl_bvec_filename is None:
        if fsl_base_filename is None:
            fsl_bvec_filename = mrtrix_filename + str(".bvec")
        else:
            fsl_bvec_filename = fsl_base_filename + str(".bvec")

    np.savetxt(fsl_bvec_filename, bvec, "%.8f")
    np.savetxt(fsl_bval_filename, bval, "%i")

    return
