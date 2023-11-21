# -*- coding: utf-8 -*-
#
# aodf := Asymmetric Orientation Distribution Function (aODF)
#

import numpy as np

from dipy.reconst.shm import sph_harm_ind_list


def compute_asymmetry_index(sh_coeffs, order, mask):
    """
    Compute asymmetry index (ASI) (Cetin-Karayumak et al, 2018) from
    asymmetric ODF volume expressed in full SH basis.

    Parameters
    ----------
    sh_coeffs: ndarray (x, y, z, ncoeffs)
         Input spherical harmonics coefficients.
    order: int > 0
         Max SH order.
    mask: ndarray (x, y, z), bool
         Mask inside which ASI should be computed.

    Returns
    -------
    asi_map: ndarray (x, y, z)
         Asymmetry index map.
    """

    _, l_list = sph_harm_ind_list(order, full_basis=True)

    sign = np.power(-1.0, l_list)
    sign = np.reshape(sign, (1, 1, 1, len(l_list)))
    sh_squared = sh_coeffs**2
    mask = np.logical_and(sh_squared.sum(axis=-1) > 0., mask)

    asi_map = np.zeros(sh_coeffs.shape[:-1])
    asi_map[mask] = np.sum(sh_squared * sign, axis=-1)[mask] / \
        np.sum(sh_squared, axis=-1)[mask]

    # Negatives should not happen (amplitudes always positive)
    asi_map = np.clip(asi_map, 0.0, 1.0)
    asi_map = np.sqrt(1 - asi_map**2) * mask

    return asi_map


def compute_odd_power_map(sh_coeffs, order, mask):
    """
    Compute odd-power map (Poirier et al, 2020) from
    asymmetric ODF volume expressed in full SH basis.

    Parameters
    ----------
    sh_coeffs: ndarray (x, y, z, ncoeffs)
         Input spherical harmonics coefficients.
    order: int > 0
         Max SH order.
    mask: ndarray (x, y, z), bool
         Mask inside which odd-power map should be computed.

    Returns
    -------
    odd_power_map: ndarray (x, y, z)
         Odd-power map.
    """
    _, l_list = sph_harm_ind_list(order, full_basis=True)
    odd_l_list = (l_list % 2 == 1).reshape((1, 1, 1, -1))

    odd_order_norm = np.linalg.norm(sh_coeffs * odd_l_list,
                                    ord=2, axis=-1)

    full_order_norm = np.linalg.norm(sh_coeffs, ord=2, axis=-1)

    asym_map = np.zeros(sh_coeffs.shape[:-1])
    mask = np.logical_and(full_order_norm > 0, mask)
    asym_map[mask] = odd_order_norm[mask] / full_order_norm[mask]

    return asym_map
