# -*- coding: utf-8 -*-
import logging

from dipy.denoise.noise_estimate import piesno
import numpy as np


def estimate_piesno_sigma(data, number_coils=0):
    """
    Here are Dipy's note on this method:
    > It is expected that
    >   1. The data has a noisy, non-masked background and
    >   2. The data is a repetition of the same measurements along the last
    >      axis, i.e. dMRI or fMRI data, not structural data like T1/T2.
    > This function processes the data slice by slice, as originally designed
    > inthe paper. Use it to get a slice by slice estimation of the noise, as
    > in spinal cord imaging for example.

    Parameters
    ----------
    data: np.ndarray
        The 4D volume.
    number_coils: int
        The number of coils in the scanner.

    Returns
    -------
    sigma: np.ndarray
        The piesno sigma estimation, one per slice.
    mask_noise: np.ndarray
        The mask of pure noise voxels inferred from the data.
    """
    assert len(data.shape) == 4

    sigma, mask_noise = piesno(data, N=number_coils, return_mask=True)

    # If the noise mask has few voxels, the detected noise standard
    # deviation can be very low and maybe something went wrong. We
    # check here that at least 1% of noisy voxels were found and warn
    # the user otherwise.
    frac_noisy_voxels = np.sum(mask_noise) / np.size(mask_noise) * 100

    if frac_noisy_voxels < 1.:
        logging.warning(
            'PIESNO was used with N={}, but it found only {:.3f}% of voxels '
            'as pure noise with a mean standard deviation of {:.5f}. This is '
            'suspicious, so please check the resulting sigma volume if '
            'something went wrong. In cases where PIESNO is not working, '
            'you might want to try basic sigma estimation.'
            .format(number_coils, frac_noisy_voxels, np.mean(sigma)))
    else:
        logging.info('The noise standard deviation from piesno is %s',
                     np.array_str(sigma[0, 0, :]))

    return sigma, mask_noise
