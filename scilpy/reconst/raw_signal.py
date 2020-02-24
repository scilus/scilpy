# -*- coding: utf-8 -*-

import logging

import numpy as np

from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh


def compute_sh_coefficients(dwi, gradient_table, sh_order=8,
                            basis_type='descoteaux07', smooth=0.006,
                            use_attenuation=False):
    """Fit a diffusion signal with spherical harmonics coefficients.

    Parameters
    ----------
    dwi : nib.Nifti1Image object
        Diffusion signal as weighted images (4D).
    gradient_table : GradientTable
        Dipy object that contains all bvals and bvecs.
    sh_order : int, optional
        SH order to fit, by default 8.
    smooth : float, optional
        Lambda-regularization coefficient in the SH fit, by default 0.006.
    basis_type: str
        Either 'tournier07' or 'descoteaux07'
    use_attenuation: bool, optional
        If true, we will use DWI attenuation. [False]

    Returns
    -------
    sh_coeffs : np.ndarray with shape (X, Y, Z, #coeffs)
        Spherical harmonics coefficients at every voxel. The actual number
        of coefficients depends on `sh_order`.
    """

    # Extracting infos
    b0_mask = gradient_table.b0s_mask
    bvecs = gradient_table.bvecs

    # Keeping b0-based infos
    bvecs = bvecs[np.logical_not(b0_mask)]
    weights = dwi[..., np.logical_not(b0_mask)]

    if use_attenuation:
        # Compute attenuation using the b0.
        b0 = dwi[..., b0_mask].mean(axis=3)
        weights = compute_dwi_attenuation(weights, b0)

    # Get cartesian coords from bvecs
    sphere = Sphere(xyz=bvecs)

    # Fit SH
    sh = sf_to_sh(weights, sphere, sh_order, basis_type, smooth)

    return sh


def compute_dwi_attenuation(dwi_weights: np.ndarray, b0: np.ndarray):
    """ Compute signal attenuation by dividing the dwi signal with the b0.

    Parameters:
    -----------
    dwi_weights : np.ndarray of shape (X, Y, Z, #gradients)
        Diffusion weighted images.
    b0 : np.ndarray of shape (X, Y, Z)
        B0 image.

    Returns
    -------
    dwi_attenuation : np.ndarray
        Signal attenuation (Diffusion weights normalized by the B0).
    """
    b0 = b0[..., None]  # Easier to work if it is a 4D array.

    # Make sure that, in every voxels, weights are lower in the b0. Should
    # always be the case, but with the noise we never know!
    erroneous_voxels = np.any(dwi_weights > b0, axis=3)
    nb_erroneous_voxels = np.sum(erroneous_voxels)
    if nb_erroneous_voxels != 0:
        logging.info("# of voxels where `dwi_signal > b0` in any direction: "
                     "{}".format(nb_erroneous_voxels))
        dwi_weights = np.minimum(dwi_weights, b0)

    # Compute attenuation
    dwi_attenuation = dwi_weights / b0
    dwi_attenuation[np.logical_not(np.isfinite(dwi_attenuation))] = 0.

    return dwi_attenuation
