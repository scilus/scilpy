# -*- coding: utf-8 -*-

import logging

from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh
import numpy as np

from scilpy.utils.bvec_bval_tools import (DEFAULT_B0_THRESHOLD,
                                          check_b0_threshold, identify_shells,
                                          is_normalized_bvecs, normalize_bvecs)


def compute_sh_coefficients(dwi, gradient_table, sh_order=4,
                            basis_type='descoteaux07', smooth=0.006,
                            use_attenuation=False, force_b0_threshold=False,
                            mask=None, sphere=None):
    """Fit a diffusion signal with spherical harmonics coefficients.

    Parameters
    ----------
    dwi : nib.Nifti1Image object
        Diffusion signal as weighted images (4D).
    gradient_table : GradientTable
        Dipy object that contains all bvals and bvecs.
    sh_order : int, optional
        SH order to fit, by default 4.
    smooth : float, optional
        Lambda-regularization coefficient in the SH fit, by default 0.006.
    basis_type: str
        Either 'tournier07' or 'descoteaux07'
    use_attenuation: bool, optional
        If true, we will use DWI attenuation. [False]
    force_b0_threshold : bool, optional
        If set, will continue even if the minimum bvalue is suspiciously high.
    mask: nib.Nifti1Image object, optional
        Binary mask. Only data inside the mask will be used for computations
        and reconstruction.
    sphere: Sphere
        Dipy object. If not provided, will use Sphere(xyz=bvecs).

    Returns
    -------
    sh_coeffs : np.ndarray with shape (X, Y, Z, #coeffs)
        Spherical harmonics coefficients at every voxel. The actual number
        of coefficients depends on `sh_order`.
    """

    # Extracting infos
    b0_mask = gradient_table.b0s_mask
    bvecs = gradient_table.bvecs
    bvals = gradient_table.bvals

    # Checks
    if not is_normalized_bvecs(bvecs):
        logging.warning("Your b-vectors do not seem normalized...")
        bvecs = normalize_bvecs(bvecs)

    b0_threshold = check_b0_threshold(force_b0_threshold, bvals.min())

    # Ensure that this is on a single shell.
    shell_values, _ = identify_shells(bvals)
    shell_values.sort()
    if shell_values.shape[0] != 2 or shell_values[0] > b0_threshold:
        raise ValueError("Can only work on single shell signals.")

    # Keeping b0-based infos
    bvecs = bvecs[np.logical_not(b0_mask)]
    weights = dwi[..., np.logical_not(b0_mask)]

    # Compute attenuation using the b0.
    if use_attenuation:
        b0 = dwi[..., b0_mask].mean(axis=3)
        weights = compute_dwi_attenuation(weights, b0)

    # Get cartesian coords from bvecs
    if sphere is None:
        sphere = Sphere(xyz=bvecs)

    # Fit SH
    sh = sf_to_sh(weights, sphere, sh_order, basis_type, smooth=smooth)

    # Apply mask
    if mask is not None:
        sh *= mask[..., None]

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

    # Make sure we didn't divide by 0.
    dwi_attenuation[np.logical_not(np.isfinite(dwi_attenuation))] = 0.

    return dwi_attenuation
