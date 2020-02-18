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


def volumes_iterator(img, size, dtype=None):
    """Generator that iterates on gradient volumes of data"""

    nb_volumes = img.shape[-1]

    if size == nb_volumes:
        yield list(range(nb_volumes)), img.get_data(dtype=dtype)
    else:
        for i in range(0, nb_volumes - size, size):
            logging.info(
                'Loading volumes {} to {}.'.format(i, i + size - 1))
            yield list(range(i, i + size)), img.dataobj[..., i:i + size]
        if i + size < nb_volumes:
            logging.info(
                'Loading volumes {} to {}.'.format(i + size, nb_volumes - 1))
            yield list(range(i + size, nb_volumes)), img.dataobj[..., i + size:]


def extract_dwi_shell(dwi_img, bvals, bvecs, bvals_to_extract, tol=10,
                      block_size=None, dtype=None):
    """
    Parameters
    ----------
    dwi_img: nib.Nifti1image of the DWI image.
    bvals: loaded bvals
    bvecs: loaded bvecs
    tol:
    block_size:
        Note. Was previously always set to img.shape[-1] if None.
    """
    # Finding the volume indices that correspond to the shells to extract.
    indices = [get_shell_indices(bvals, shell, tol=tol)
               for shell in bvals_to_extract]
    indices = np.unique(np.sort(np.hstack(indices)))
    if len(indices) == 0:
        raise ValueError('There are no volumes that have the supplied '
                         'b-values.')

    if block_size is None:
        # Using the easy way. Faster.
        shell_data = dwi_img.get_fdata(dtype=dtype)[..., indices]
    else:
        # Loading the shells by iterating through blocks of volumes.
        # This approach is slower for small files, but allows very big files to
        # be split with less memory usage.

        shell_data = np.zeros((dwi_img.shape[:-1] + (len(indices),)))
        for vi, data in volumes_iterator(dwi_img, block_size, dtype):
            in_volume = np.array([i in vi for i in indices])
            in_data = np.array([i in indices for i in vi])
            shell_data[..., in_volume] = data[..., in_data]

    bvals = bvals[indices].astype(int)
    bvals.shape = (1, len(bvals))

    bvecs = bvecs[indices, :]

    return shell_data, bvals, bvecs, indices


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
