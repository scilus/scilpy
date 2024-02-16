# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_to_fwhm(data, fwhm):
    """
    Smooth a volume to given FWHM.

    Parameters
    ----------
    data: np.ndarray
        3D or 4D data. If it is 4D, processing invidually on each volume (on
        the last dimension)
    fwhm: float
        Full width at half maximum.
    """
    if fwhm > 0:
        # converting fwhm to Gaussian std
        gauss_std = fwhm / np.sqrt(8 * np.log(2))

        if len(data.shape) == 3:
            data_smooth = gaussian_filter(data, sigma=gauss_std)
        elif  len(data.shape) == 4:
            data_smooth = np.zeros(data.shape)
            for v in range(data.shape[-1]):
                data_smooth[..., v] = gaussian_filter(data[..., v],
                                                      sigma=gauss_std)
        else:
            raise ValueError("Expecting a 3D or 4D volume.")

        return data_smooth
    else:
        return data
