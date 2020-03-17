# -*- coding: utf-8 -*-

import logging

import numpy as np

from scilpy.utils.filenames import split_name_with_nii


def save_scheme_mrtrix(points, shell_idx, bvals, filename):
    """
    Save table gradient (MRtrix format)

    Parameters
    ----------
    points: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs in points.
    bvals: numpy.array
    filename: str
        output file name
    ------
    """
    with open(filename, 'w') as f:
        for idx in range(points.shape[0]):
            f.write('{:.8f} {:.8f} {:.8f} {:.2f}\n'.format(points[idx, 0],
                                                           points[idx, 1],
                                                           points[idx, 2],
                                                           bvals[shell_idx[idx]]))

    logging.info('Scheme saved in MRtrix format as {}'.format(filename))


def save_scheme_bvecs_bvals(points, shell_idx, bvals, filename_bval,
                            filename_bvec):
    """
    Save table gradient (FSL format)

    Parameters
    ----------
    points: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs in points.
    bvals: numpy.array
    filename_bval: str
        output bval filename.
    filename_bvec: str
        output bvec filename.
    ------
    """
    fullfilename, ext = split_name_with_nii(filename_bval)

    np.savetxt(filename_bvec, points.T, fmt='%.8f')
    np.savetxt(filename_bval, np.array([bvals[idx] for idx in shell_idx])[None, :], fmt='%.3f')

    logging.info('Scheme saved in FSL format as {}'.format(fullfilename +
                                                           '{.bvec/.bval}'))
