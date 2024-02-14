# -*- coding: utf-8 -*-

import logging
import os

import numpy as np


def fsl2mrtrix(fsl_bval_filename, fsl_bvec_filename, mrtrix_filename):
    """
    Convert a fsl dir_grad.bvec/.bval files to mrtrix encoding.b file.
    Saves the result.

    Parameters
    ----------
    fsl_bval_filename: str
        path to input fsl bval file.
    fsl_bvec_filename: str
        path to input fsl bvec file.
    mrtrix_filename : str
        path to output mrtrix encoding.b file.
    """
    shells = np.loadtxt(fsl_bval_filename)
    points = np.loadtxt(fsl_bvec_filename)
    bvals = np.unique(shells).tolist()

    # Remove .bval and .bvec if present
    mrtrix_filename = mrtrix_filename.replace('.b', '')

    if not points.shape[0] == 3:
        points = points.transpose()
        logging.warning('WARNING: Your bvecs seem transposed. ' +
                        'Transposing them.')

    shell_idx = [int(np.where(bval == bvals)[0]) for bval in shells]
    save_gradient_sampling_mrtrix(points, shell_idx, bvals,
                                  mrtrix_filename + '.b')


def mrtrix2fsl(mrtrix_filename, fsl_filename):
    """
    Convert a mrtrix encoding.b file to fsl dir_grad.bvec/.bval files.
    Saves the result.

    Parameters
    ----------
    mrtrix_filename : str
        path to mrtrix encoding.b file.
    fsl_filename: str
        path to the output fsl files. Files will be named
        fsl_bval_filename.bval and fsl_bval_filename.bvec.
    """
    # Remove .bval and .bvec if present
    fsl_filename = fsl_filename.replace('.bval', '')
    fsl_filename = fsl_filename.replace('.bvec', '')

    mrtrix_b = np.loadtxt(mrtrix_filename)
    if not len(mrtrix_b.shape) == 2 or not mrtrix_b.shape[1] == 4:
        raise ValueError('mrtrix file must have 4 columns')

    points = np.array([mrtrix_b[:, 0], mrtrix_b[:, 1], mrtrix_b[:, 2]])
    shells = np.array(mrtrix_b[:, 3])

    bvals = np.unique(shells).tolist()
    shell_idx = [int(np.where(bval == bvals)[0]) for bval in shells]

    save_gradient_sampling_fsl(points, shell_idx, bvals,
                               filename_bval=fsl_filename + '.bval',
                               filename_bvec=fsl_filename + '.bvec')


def save_gradient_sampling_mrtrix(bvecs, shell_idx, bvals, filename):
    """
    Save table gradient (MRtrix format)

    Parameters
    ----------
    bvecs: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs.
    bvals: numpy.array
    filename: str
        output file name
    ------
    """
    with open(filename, 'w') as f:
        for idx in range(bvecs.shape[1]):
            f.write('{:.8f} {:.8f} {:.8f} {:}\n'
                    .format(bvecs[0, idx], bvecs[1, idx], bvecs[2, idx],
                            bvals[shell_idx[idx]]))

    logging.info('Gradient sampling saved in MRtrix format as {}'
                 .format(filename))


def save_gradient_sampling_fsl(bvecs, shell_idx, bvals, filename_bval,
                               filename_bvec):
    """
    Save table gradient (FSL format)

    Parameters
    ----------
    bvecs: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs.
    bvals: numpy.array
    filename_bval: str
        output bval filename.
    filename_bvec: str
        output bvec filename.
    ------
    """
    basename, ext = os.path.splitext(filename_bval)

    np.savetxt(filename_bvec, bvecs, fmt='%.8f')
    np.savetxt(filename_bval,
               np.array([bvals[idx] for idx in shell_idx])[None, :],
               fmt='%.3f')

    logging.info('Gradient sampling saved in FSL format as {}'
                 .format(basename + '{.bvec/.bval}'))
