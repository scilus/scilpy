# -*- coding: utf-8 -*-

from dipy.direction.peaks import peak_directions
import numpy as np


def find_order_from_nb_coeff(data):
    if isinstance(data, np.ndarray):
        shape = data.shape
    else:
        shape = data
    return int((-3 + np.sqrt(1 + 8 * shape[-1])) / 2)


def get_sh_order_and_fullness(ncoeffs):
    """
    Get the order of the SH basis from the number of SH coefficients
    as well as a boolean indicating if the basis is full.
    """
    # the two curves (sym and full) intersect at ncoeffs = 1, in what
    # case both bases correspond to order 1.
    sym_order = (-3.0 + np.sqrt(1.0 + 8.0 * ncoeffs)) / 2.0
    if sym_order.is_integer():
        return sym_order, False
    full_order = np.sqrt(ncoeffs) - 1.0
    if full_order.is_integer():
        return full_order, True
    raise ValueError('Invalid number of coefficients for SH basis.')


def get_maximas(data, sphere, b_matrix, threshold, absolute_threshold,
                min_separation_angle=25):
    spherical_func = np.dot(data, b_matrix.T)
    spherical_func[np.nonzero(spherical_func < absolute_threshold)] = 0.
    return peak_directions(
        spherical_func, sphere, threshold, min_separation_angle)


def get_sphere_neighbours(sphere, max_angle):
    """
    Get a matrix of neighbours for each direction on the sphere, within
    the min_separation_angle.

    min_separation_angle: float
        Maximum angle in radians defining the neighbourhood
        of each direction.

    Return
    ------
    neighbours: ndarray
        Neighbour directions for each direction on the sphere.
    """
    xs = sphere.vertices[:, 0]
    ys = sphere.vertices[:, 1]
    zs = sphere.vertices[:, 2]
    scalar_prods = (np.outer(xs, xs) + np.outer(ys, ys) +
                    np.outer(zs, zs))
    neighbours = scalar_prods >= np.cos(max_angle)
    return neighbours

def is_data_peaks(img_data):
    """
    Heuristic to find out if the input are peaks or fodf.
    fodf are always around 0.15 and peaks around 0.75.
    Peaks have more zero values than fodf. The first value of fodf is
    usually the highest.

    Parameters
    ----------
    img_data : np.ndarray
        4D image data where the last dimension contains directional info.

    Returns
    -------
    is_peaks : bool
        True if data is likely peaks, False if likely fODF (SH).
    """
    last_dim = img_data.shape[-1]
    if last_dim == 3:
        return True

    # Sum of absolute values to detect non-zero voxels correctly
    non_zeros_mask = np.any(np.abs(img_data) > 0, axis=-1)
    if not np.count_nonzero(non_zeros_mask):
        return False

    try:
        from scilpy.reconst.utils import get_sh_order_and_fullness
        order, full = get_sh_order_and_fullness(last_dim)
        # Symmetric SH must be even order
        if not full and order % 2 != 0:
            return False
    except ValueError:
        # If not a valid SH number of coefficients, and not 3,
        # it might be something else, but if it's a multiple of 3
        # it's likely Peaks.
        if last_dim % 3 == 0:
            return True
        return False

    data_nz = img_data[non_zeros_mask]

    # If all triplets have the same norm, it is likely peaks, otherwise SH.
    if last_dim % 3 == 0:
        norm = np.linalg.norm(data_nz.reshape(-1, 3), axis=-1)
        if np.all(np.isclose(norm, norm[0])):
            return True

    # If the max is in the first triplet but not at index 0, it's likely Peaks.
    # Smoothed SH almost always has max at index 0
    argmax_indices = np.argmax(np.abs(data_nz), axis=-1)
    if last_dim % 3 == 0 and \
            np.mean(np.logical_or(argmax_indices == 1,
                                  argmax_indices == 2)) > 0.1:
        return True

    # Exact zeros. SH almost never has exact zeros in real data.
    # Peaks often have exact zeros for unused lobes
    zero_ratio = np.mean(data_nz == 0)
    if zero_ratio > 0.05:
        return True

    # Default to SH
    return False
