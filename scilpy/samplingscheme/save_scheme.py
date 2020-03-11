# -*- coding: utf-8 -*-

import logging
import numpy as np

from scilpy.utils.filenames import split_name_with_nii


def save_scheme_caru(points, shell_idx, filename):
    """
    Save gradient table (Caruyer format)

    Parameters
    ----------
    points: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs in points.
    filename: output file name
    ------
    """

    fullfilename, ext = split_name_with_nii(filename)
    fullfilename = fullfilename + '.caru'

    with open(fullfilename) as f:
        f.write('# Caruyer format sampling scheme\n')
        f.write('# X Y Z shell_idx\n')
        for idx in range(points.shape[0]):
            f.write('{:.8f} {:.8f} {:.8f} {:.0f}\n'.format(points[idx, 0],
                                                           points[idx, 1],
                                                           points[idx, 2],
                                                           shell_idx[idx]))

    logging.info('Scheme saved in Caruyer format as {}'.format(fullfilename))


def save_scheme_philips(points, shell_idx, bvals, filename):
    """
    Save table gradient (Philips format)

    Parameters
    ----------
    points: numpy.array
        bvecs normalized to 1
    shell_idx: numpy.array
        Shell index for bvecs in points.
    bvals: numpy.array
    filename: str
        output file name
    ------
    """
    fullfilename, ext = split_name_with_nii(filename)
    fullfilename = fullfilename + '.txt'

    with open(fullfilename) as f:
        f.write('# Philips format sampling scheme\n')
        f.write('# X Y Z bval\n')
        for idx in range(points.shape[0]):
            f.write('{:.3f} {:.3f} {:.3f} {:.2f}\n'.format(points[idx, 0],
                                                           points[idx, 1],
                                                           points[idx, 2],
                                                           bvals[shell_idx[idx]]))

    logging.info('Scheme saved in Philips format as {}'.format(fullfilename))


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
    fullfilename, ext = split_name_with_nii(filename)
    fullfilename = fullfilename + '.b'

    with open(fullfilename) as f:
        for idx in range(points.shape[0]):
            f.write('{:.8f} {:.8f} {:.8f} {:.2f}\n'.format(points[idx, 0],
                                                           points[idx, 1],
                                                           points[idx, 2],
                                                           bvals[shell_idx[idx]]))

    logging.info('Scheme saved in MRtrix format as {}'.format(fullfilename))


def save_scheme_bvecs_bvals(points, shell_idx, bvals, filename=None,
                            filename_bval=None, filename_bvec=None):
    """
    Save table gradient (FSL format)

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
    if filename:
        fullfilename, ext = split_name_with_nii(filename)
        filename_bval = fullfilename + '.bval'
        filename_bvec = fullfilename + '.bvec'

    np.savetxt(filename_bvec, points.T, fmt='%.8f')
    np.savetxt(filename_bval, np.array([bvals[idx] for idx in shell_idx])[None, :], fmt='%.3f')

    logging.info('Scheme saved in FSL format as {}'.format(fullfilename +
                                                           '{.bvec/.bval}'))


def save_scheme_siemens(points, shell_idx, bvals, filename):
    """
    Save table gradient (Siemens format)

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

    str_save = []
    str_save.append('[Directions={}]'.format(points.shape[0]))
    str_save.append('CoordinateSystem = XYZ')
    str_save.append('Normalisation = None')

    # Scale bvecs with q-value
    bvals = np.array([bvals[idx] for idx in shell_idx])
    bmax = np.array(bvals).max()
    bvecs_norm = (bvals / float(bmax))**(0.5)

    # ugly work around for the division by b0 / replacing NaNs with 0.0
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    points = points / bvecs_norm[:, None]
    np.seterr(**old_settings)
    points[np.isnan(points)] = 0.0
    points[np.isinf(points)] = 0.0

    for idx in range(points.shape[0]):
        str_save.append('vector[{}] = ( {}, {}, {} )'.format(idx,
                                                             points[idx, 0],
                                                             points[idx, 1],
                                                             points[idx, 2]))

    fullfilename, ext = split_name_with_nii(filename)
    fullfilename = fullfilename + '.dvs'
    with open(fullfilename) as f:
        for idx in range(len(str_save)):
            f.write(str_save[idx] + '\n')

    logging.info('Scheme saved in Siemens format as {}'.format(fullfilename))
