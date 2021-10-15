# -*- coding: utf-8 -*-

import numpy as np
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from dipy.core.sphere import Sphere
from scipy.stats import multivariate_normal


def multivariate_bilateral_filtering(in_sh, sh_order=8,
                                     sh_basis='descoteaux07',
                                     in_full_basis=False,
                                     out_full_basis=True,
                                     sphere_str='repulsion724',
                                     var_cov=np.eye(2),
                                     sigma_range=0.5):
    """Average the SH projected on a sphere using a first-neighbor gaussian
    blur and a dot product weight between sphere directions and the direction
    to neighborhood voxels, forcing to 0 negative values and thus performing
    asymmetric hemisphere-aware filtering.

    Parameters
    ----------
    in_sh: ndarray (x, y, z, n_coeffs)
        Input SH coefficients array
    sh_order: int, optional
        Maximum order of the SH series.
    sh_basis: {'descoteaux07', 'tournier07'}, optional
        SH basis of the input signal.
    in_full_basis: bool, optional
        True if the input is in full SH basis.
    out_full_basis: bool, optional
        If True, save output SH using full SH basis.
    dot_sharpness: float, optional
        Exponent of the dot product. When set to 0.0, directions
        are not weighted by the dot product.
    sphere_str: str, optional
        Name of the sphere used to project SH coefficients to SF.
    sigma_spatial: float, optional
        Sigma for the Gaussian.
    sigma_range: float, optional

    Returns
    -------
    out_sh: ndarray (x, y, z, n_coeffs)
        Filtered signal as SH coefficients.
    """
    # Load the sphere used for projection of SH
    sphere = get_sphere(sphere_str)

    # Normalized filter for each sf direction
    weights = _get_weights_multivariate(sphere, var_cov)

    nb_sf = len(sphere.vertices)
    mean_sf = np.zeros(np.append(in_sh.shape[:-1], nb_sf))
    B = sh_to_sf_matrix(sphere, sh_order=sh_order, basis_type=sh_basis,
                        return_inv=False, full_basis=in_full_basis)

    # We want a B matrix to project on an inverse sphere to have the sf on
    # the opposite hemisphere for a given vertice
    neg_B = sh_to_sf_matrix(Sphere(xyz=-sphere.vertices), sh_order=sh_order,
                            basis_type=sh_basis, return_inv=False,
                            full_basis=in_full_basis)

    # Apply filter to each sphere vertice
    for sf_i in range(nb_sf):
        w_filter = weights[..., sf_i]

        # Generate 1-channel images for directions u and -u
        current_sf = np.dot(in_sh, B[:, sf_i])
        opposite_sf = np.dot(in_sh, neg_B[:, sf_i])

        mean_sf[..., sf_i] = correlate_spatial(current_sf,
                                               opposite_sf,
                                               w_filter,
                                               sigma_range)

    # Convert back to SH coefficients
    _, B_inv = sh_to_sf_matrix(sphere, sh_order=sh_order, basis_type=sh_basis,
                               full_basis=out_full_basis)
    out_sh = np.array([np.dot(i, B_inv) for i in mean_sf], dtype=in_sh.dtype)
    return out_sh


def evaluate_gaussian_dist(x, sigma):
    assert sigma > 0.0, "Sigma must be greater than 0."
    cnorm = 1.0 / sigma / np.sqrt(2.0*np.pi)
    return cnorm * np.exp(-x**2/2/sigma**2)


def evaluate_multivariate_gaussian(x, cov):
    x_shape = x.shape
    x = x.reshape((-1, 2))
    flat_out = multivariate_normal.pdf(x, mean=[0, 0], cov=cov)

    fx = flat_out.reshape(x_shape[:-1])

    return fx


def _get_weights_multivariate(sphere, cov):
    """
    Get neighbors weight in respect to the direction to a voxel.

    Parameters
    ----------
    sphere: Sphere
        Sphere used for SF reconstruction.
    dot_sharpness: float
        Dot product exponent.
    sigma: float
        Variance of the gaussian used for weighting neighbors.

    Returns
    -------
    weights: ndarray
        Vertices weights with respect to voxel directions.
    """
    win_size = np.ceil(6.0 * cov[0, 0] + 1.0).astype(int)
    if win_size % 2 == 0:
        win_size += 1
    half_size = win_size // 2
    directions = np.zeros((win_size, win_size, win_size, 3))
    for x in range(win_size):
        for y in range(win_size):
            for z in range(win_size):
                directions[x, y, z, 0] = x - half_size
                directions[x, y, z, 1] = y - half_size
                directions[x, y, z, 2] = z - half_size

    non_zero_dir = np.ones(directions.shape[:-1], dtype=bool)
    non_zero_dir[half_size, half_size, half_size] = False

    # normalize dir
    dir_norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions[non_zero_dir] /= dir_norm[non_zero_dir]

    # angle in the range [0, pi]
    angle = np.arccos(np.dot(directions, sphere.vertices.T))
    angle[half_size, half_size, half_size, :] = 0.0

    dir_norm = np.broadcast_to(dir_norm, angle.shape)
    norm_angles = np.stack([dir_norm, angle], axis=-1)

    weights = evaluate_multivariate_gaussian(norm_angles, cov)

    weights /= weights.reshape((-1, weights.shape[-1])).sum(axis=0)

    return weights


def correlate_spatial(image_u, image_neg_u, h_filter, sigma_range):
    """
    Implementation of correlate function.
    """
    h_w, h_h, h_d = h_filter.shape[:3]
    half_w, half_h, half_d = h_w // 2, h_h // 2, h_d // 2
    pad_img = np.zeros((image_neg_u.shape[0] + 2*half_w,
                        image_neg_u.shape[1] + 2*half_h,
                        image_neg_u.shape[2] + 2*half_d))
    pad_img[half_w:-half_w, half_h:-half_h, half_d:-half_d] = image_neg_u

    out_im = np.zeros_like(image_u)
    for ii in range(image_u.shape[0]):
        for jj in range(image_u.shape[1]):
            for kk in range(image_u.shape[2]):
                x = pad_img[ii:ii+h_w, jj:jj+h_h, kk:kk+h_d]\
                    - image_u[ii, jj, kk]
                range_filter = evaluate_gaussian_dist(x, sigma_range)

                res_filter = range_filter * h_filter
                # Divide the filter into two filters;
                # One for the current sphere direction and
                # the other for the opposite direction.
                res_filter_sum = np.sum(res_filter)
                center_val = res_filter[half_w, half_h, half_d]
                res_filter[half_w, half_h, half_d] = 0.0

                out_im[ii, jj, kk] = image_u[ii, jj, kk] * center_val
                out_im[ii, jj, kk] += np.sum(pad_img[ii:ii+h_w,
                                                     jj:jj+h_h,
                                                     kk:kk+h_d]
                                             * res_filter)
                out_im[ii, jj, kk] /= res_filter_sum

    return out_im
