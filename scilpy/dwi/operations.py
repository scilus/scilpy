import logging
import numpy as np


def apply_bias_field(dwi_data, bias_field_data, mask_data):
    """
    ToDo: Explain formula why applying field = dividing?
     + why we need to rescale after?

    Parameters
    ----------
    dwi_data: np.ndarray
        The 4D dwi data.
    bias_field_data: np.ndarray
        The 3D bias field.
    mask_data: np.ndarray
        The mask where to apply the bias field.

    Returns
    -------
    dwi_data: np.ndarray
        The modified 4D dwi_data.
    """
    nuc_dwi_data = np.divide(
        dwi_data[mask_data],
        bias_field_data[mask_data].reshape((len(mask_data[0]), 1)))

    rescaled_nuc_data = _rescale_dwi(dwi_data[mask_data], nuc_dwi_data)
    dwi_data[mask_data] = rescaled_nuc_data

    return dwi_data


def _rescale_intensity(val, slope, in_max, bc_max):
    """
    Rescale an intensity value given a scaling factor.
    This scaling factor ensures that the intensity
    range before and after correction is the same.

    Parameters
    ----------
    val: float
         Value to be scaled
    scale: float
         Scaling factor to be applied
    in_max: float
         Max possible value
    bc_max: float
         Max value in the bias correction value range

    Returns
    -------
    rescaled_value: float
         Bias field corrected value scaled by the slope
         of the data
    """

    return in_max - slope * (bc_max - val)


# https://github.com/stnava/ANTs/blob/master/Examples/N4BiasFieldCorrection.cxx
def _rescale_dwi(in_data, bc_data):
    """
    Apply N4 Bias Field Correction to a DWI volume.
    bc stands for bias correction. The code comes
    from the C++ ANTS implmentation.

    Parameters
    ----------
    in_data: ndarray (x, y, z, ndwi)
         Input DWI volume 4-dimensional data.
    bc_data: ndarray (x, y, z, ndwi)
         Bias field correction volume estimated from ANTS
         Copied for every dimension of the DWI 4-th dimension

    Returns
    -------
    bc_data: ndarray (x, y, z, ndwi)
         Bias field corrected DWI volume
    """

    in_min = np.amin(in_data)
    in_max = np.amax(in_data)
    bc_min = np.amin(bc_data)
    bc_max = np.amax(bc_data)

    slope = (in_max - in_min) / (bc_max - bc_min)

    chunk = np.arange(0, len(in_data), 100000)
    chunk = np.append(chunk, len(in_data))
    for i in range(len(chunk)-1):
        nz_bc_data = bc_data[chunk[i]:chunk[i+1]]
        rescale_func = np.vectorize(_rescale_intensity, otypes=[np.float32])

        rescaled_data = rescale_func(nz_bc_data, slope, in_max, bc_max)
        bc_data[chunk[i]:chunk[i+1]] = rescaled_data

    return bc_data


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
