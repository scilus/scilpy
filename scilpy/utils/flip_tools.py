# -*- coding: utf-8 -*-

import numpy as np


def flip_mrtrix_gradient_sampling(gradient_sampling_filename,
                                  gradient_sampling_flipped_filename, axes):
    """
    Flip Mrtrix gradient sampling on a axis

    Parameters
    ----------
    gradient_sampling_filename: str
        Gradient sampling filename
    gradient_sampling_flipped_filename: str
        Gradient sampling flipped filename
    axes: list of int
        List of axes to flip (e.g. [0, 1])
    """
    gradient_sampling = np.loadtxt(gradient_sampling_filename)
    for axis in axes:
        gradient_sampling[:, axis] *= -1

    np.savetxt(gradient_sampling_flipped_filename,
               gradient_sampling,
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
