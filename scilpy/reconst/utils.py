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
