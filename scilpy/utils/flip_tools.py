# -*- coding: utf-8 -*-

import numpy as np


def flip_mrtrix_gradient_sampling(encoding_scheme_filename,
                                  encoding_scheme_flipped_filename, axes):
    """
    Flip Mrtrix gradient sampling on a axis

    Parameters
    ----------
    encoding_scheme_filename: str
        Encoding scheme filename
    encoding_scheme_flipped_filename: str
        Encoding scheme flipped filename
    axes: list of int
        List of axes to flip (e.g. [0, 1])
    """
    encoding_scheme = np.loadtxt(encoding_scheme_filename)
    for axis in axes:
        encoding_scheme[:, axis] *= -1

    np.savetxt(encoding_scheme_flipped_filename,
               encoding_scheme,
               "%.8f %.8f %.8f %0.6f")


def flip_fsl_gradient_sampling(bvecs_filename, bvecs_flipped_filename, axes):
    """
    Flip FSL bvecs on a axis

    Parameters
    ----------
    bvecs_filename: str
        Bvecs filename
    bvecs_flipped_filename: str
        Bvecs flipped filename
    axes: list of int
        List of axes to flip (e.g. [0, 1])
    """
    bvecs = np.loadtxt(bvecs_filename)
    for axis in axes:
        bvecs[axis, :] *= -1

    np.savetxt(bvecs_flipped_filename, bvecs, "%.8f")
