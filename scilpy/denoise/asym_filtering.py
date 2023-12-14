# -*- coding: utf-8 -*-

import numpy as np
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from dipy.core.sphere import Sphere
from scipy.ndimage import correlate
from scilpy.gpuparallel.opencl_utils import have_opencl, CLKernel, CLManager


def angle_aware_bilateral_filtering(in_sh, sh_order=8,
                                    sh_basis='descoteaux07',
                                    in_full_basis=False,
                                    sphere_str='repulsion724',
                                    sigma_spatial=1.0, sigma_angular=1.0,
                                    sigma_range=0.5, use_gpu=True):
    """
    Angle-aware bilateral filtering.

    Parameters
    ----------
    in_sh: ndarray (x, y, z, ncoeffs)
        Input SH volume.
    sh_order: int, optional
        Maximum SH order of input volume.
    sh_basis: str, optional
        Name of SH basis used.
    in_full_basis: bool, optional
        True if input is expressed in full SH basis.
    sphere_str: str, optional
        Name of the DIPY sphere to use for sh to sf projection.
    sigma_spatial: float, optional
        Standard deviation for spatial filter.
    sigma_angular: float, optional
        Standard deviation for angular filter.
    sigma_range: float, optional
        Standard deviation for range filter.
    use_gpu: bool, optional
        True if GPU should be used.

    Returns
    -------
    out_sh: ndarray (x, y, z, ncoeffs)
        Output SH coefficient array in full SH basis.
    """
    if use_gpu and have_opencl:
        return angle_aware_bilateral_filtering_gpu(in_sh, sh_order,
                                                   sh_basis, in_full_basis,
                                                   sphere_str, sigma_spatial,
                                                   sigma_angular, sigma_range)
    elif use_gpu and not have_opencl:
        raise RuntimeError('Package pyopencl not found. Install pyopencl'
                           ' or set use_gpu to False.')
    else:
        return angle_aware_bilateral_filtering_cpu(in_sh, sh_order,
                                                   sh_basis, in_full_basis,
                                                   sphere_str, sigma_spatial,
                                                   sigma_angular, sigma_range)


def angle_aware_bilateral_filtering_gpu(in_sh, sh_order=8,
                                        sh_basis='descoteaux07',
                                        in_full_basis=False,
                                        sphere_str='repulsion724',
                                        sigma_spatial=1.0,
                                        sigma_angular=1.0,
                                        sigma_range=0.5):
    """
    Angle-aware bilateral filtering using OpenCL for GPU computing.

    Parameters
    ----------
    in_sh: ndarray (x, y, z, ncoeffs)
        Input SH volume.
    sh_order: int, optional
        Maximum SH order of input volume.
    sh_basis: str, optional
        Name of SH basis used.
    in_full_basis: bool, optional
        True if input is expressed in full SH basis.
    sphere_str: str, optional
        Name of the DIPY sphere to use for sh to sf projection.
    sigma_spatial: float, optional
        Standard deviation for spatial filter.
    sigma_angular: float, optional
        Standard deviation for angular filter.
    sigma_range: float, optional
        Standard deviation for range filter.

    Returns
    -------
    out_sh: ndarray (x, y, z, ncoeffs)
        Output SH coefficient array in full SH basis.
    """
    s_weights = _get_spatial_weights(sigma_spatial)
    h_half_width = len(s_weights) // 2

    sphere = get_sphere(sphere_str)
    a_weights = _get_angular_weights(s_weights.shape, sphere, sigma_angular)

    h_weights = s_weights[..., None] * a_weights
    h_weights /= np.sum(h_weights, axis=(0, 1, 2))

    sh_to_sf_mat = sh_to_sf_matrix(sphere, sh_order=sh_order,
                                   basis_type=sh_basis,
                                   full_basis=in_full_basis,
                                   return_inv=False)

    _, sf_to_sh_mat = sh_to_sf_matrix(sphere, sh_order=sh_order,
                                      basis_type=sh_basis,
                                      full_basis=True,
                                      return_inv=True)

    out_n_coeffs = sf_to_sh_mat.shape[1]
    n_dirs = len(sphere.vertices)
    volume_shape = in_sh.shape
    in_sh = np.pad(in_sh, ((h_half_width, h_half_width),
                           (h_half_width, h_half_width),
                           (h_half_width, h_half_width),
                           (0, 0)))

    cl_kernel = CLKernel('correlate', 'denoise', 'angle_aware_bilateral.cl')
    cl_kernel.set_define('IM_X_DIM', volume_shape[0])
    cl_kernel.set_define('IM_Y_DIM', volume_shape[1])
    cl_kernel.set_define('IM_Z_DIM', volume_shape[2])

    cl_kernel.set_define('H_X_DIM', h_weights.shape[0])
    cl_kernel.set_define('H_Y_DIM', h_weights.shape[1])
    cl_kernel.set_define('H_Z_DIM', h_weights.shape[2])

    cl_kernel.set_define('SIGMA_RANGE', float(sigma_range))

    cl_kernel.set_define('IN_N_COEFFS', volume_shape[-1])
    cl_kernel.set_define('OUT_N_COEFFS', out_n_coeffs)
    cl_kernel.set_define('N_DIRS', n_dirs)

    cl_manager = CLManager(cl_kernel, 4, 1)
    cl_manager.add_input_buffer(0, in_sh)
    cl_manager.add_input_buffer(1, h_weights)
    cl_manager.add_input_buffer(2, sh_to_sf_mat)
    cl_manager.add_input_buffer(3, sf_to_sh_mat)

    cl_manager.add_output_buffer(0, volume_shape[:3] + (out_n_coeffs,),
                                 np.float32)

    outputs = cl_manager.run(volume_shape[:3])
    return outputs[0]


def angle_aware_bilateral_filtering_cpu(in_sh, sh_order=8,
                                        sh_basis='descoteaux07',
                                        in_full_basis=False,
                                        sphere_str='repulsion724',
                                        sigma_spatial=1.0,
                                        sigma_angular=1.0,
                                        sigma_range=0.5):
    """
    Angle-aware bilateral filtering on the CPU
    (optionally using multiple threads).

    Parameters
    ----------
    in_sh: ndarray (x, y, z, ncoeffs)
        Input SH volume.
    sh_order: int, optional
        Maximum SH order of input volume.
    sh_basis: str, optional
        Name of SH basis used.
    in_full_basis: bool, optional
        True if input is expressed in full SH basis.
    sphere_str: str, optional
        Name of the DIPY sphere to use for sh to sf projection.
    sigma_spatial: float, optional
        Standard deviation for spatial filter.
    sigma_angular: float, optional
        Standard deviation for angular filter.
    sigma_range: float, optional
        Standard deviation for range filter.

    Returns
    -------
    out_sh: ndarray (x, y, z, ncoeffs)
        Output SH coefficient array in full SH basis.
    """
    # Load the sphere used for projection of SH
    sphere = get_sphere(sphere_str)

    # Normalized filter for each sf direction
    s_weights = _get_spatial_weights(sigma_spatial)
    a_weights = _get_angular_weights(s_weights.shape, sphere, sigma_angular)

    weights = s_weights[..., None] * a_weights
    weights /= np.sum(weights, axis=(0, 1, 2))

    nb_sf = len(sphere.vertices)
    B = sh_to_sf_matrix(sphere, sh_order=sh_order, basis_type=sh_basis,
                        return_inv=False, full_basis=in_full_basis)

    mean_sf = np.zeros(in_sh.shape[:-1] + (nb_sf,))

    # Apply filter to each sphere vertice
    for sph_id in range(nb_sf):
        w_filter = weights[..., sph_id]

        # Generate 1-channel images for directions u and -u
        current_sf = np.dot(in_sh, B[:, sph_id])
        mean_sf[..., sph_id] = _correlate_spatial(current_sf, w_filter,
                                                  sigma_range)

    # Convert back to SH coefficients
    _, B_inv = sh_to_sf_matrix(sphere, sh_order=sh_order, basis_type=sh_basis,
                               full_basis=True)
    out_sh = np.array([np.dot(i, B_inv) for i in mean_sf], dtype=in_sh.dtype)
    # By default, return only asymmetric SH
    return out_sh


def _evaluate_gaussian_distribution(x, sigma):
    """
    1-dimensional 0-centered Gaussian distribution
    with standard deviation sigma.

    Parameters
    ----------
    x: ndarray or float
        Points where the distribution is evaluated.
    sigma: float
        Standard deviation.

    Returns
    -------
    out: ndarray or float
        Values at x.
    """
    assert sigma > 0.0, "Sigma must be greater than 0."
    cnorm = 1.0 / sigma / np.sqrt(2.0*np.pi)
    return cnorm * np.exp(-x**2/2/sigma**2)


def _get_window_directions(shape):
    """
    Get directions from center voxel to all neighbours
    for a window of given shape.

    Parameters
    ----------
    shape: tuple
        Dimensions of the window.

    Returns
    -------
    grid: ndarray
        Grid containing the direction from the center voxel to
        the current position for all positions inside the window.
    """
    grid = np.indices(shape)
    grid = np.moveaxis(grid, 0, -1)
    grid = grid - np.asarray(shape) // 2
    return grid


def _get_spatial_weights(sigma_spatial):
    """
    Compute the spatial filter, which is an isotropic Gaussian filter
    of standard deviation sigma_spatial forweighting by the distance
    between voxel positions. The size of the filter is given by
    6 * sigma_spatial, in order to cover the range
    [-3*sigma_spatial, 3*sigma_spatial].

    Parameters
    ----------
    sigma_spatial: float
        Standard deviation of spatial filter.

    Returns
    -------
    spatial_weights: ndarray
        Spatial filter.
    """
    shape = int(6 * sigma_spatial)
    if shape % 2 == 0:
        shape += 1
    shape = (shape, shape, shape)

    grid = _get_window_directions(shape)

    distances = np.linalg.norm(grid, axis=-1)
    spatial_weights = _evaluate_gaussian_distribution(distances, sigma_spatial)

    # normalize filter
    spatial_weights /= np.sum(spatial_weights)
    return spatial_weights


def _get_angular_weights(shape, sphere, sigma_angular):
    """
    Compute the angular filter, weighted by the alignment between a
    sphere direction and the direction to a neighbour. The parameter
    sigma_angular controls the sharpness of the kernel.

    Parameters
    ----------
    shape: tuple
        Shape of the angular filter.
    sphere: dipy Sphere
        Sphere on which the SH coefficeints are projected.
    sigma_angular: float
        Standard deviation of Gaussian distribution.

    Returns
    -------
    angular_weights: ndarray
        Angular filter for each position and for each sphere direction.
    """
    grid_dirs = _get_window_directions(shape).astype(np.float32)
    dir_norms = np.linalg.norm(grid_dirs, axis=-1)

    # normalized grid directions
    grid_dirs[dir_norms > 0] /= dir_norms[dir_norms > 0][:, None]
    angles = np.arccos(np.dot(grid_dirs, sphere.vertices.T))
    angles[np.logical_not(dir_norms > 0), :] = 0.0

    angular_weights = _evaluate_gaussian_distribution(angles, sigma_angular)

    # normalize filter per direction
    angular_weights /= np.sum(angular_weights, axis=(0, 1, 2))
    return angular_weights


def _correlate_spatial(image_u, h_filter, sigma_range):
    """
    Implementation of the correlation operation for anisotropic filtering.

    Parameters
    ----------
    image_u: ndarray (X, Y, Z)
        SF image for some sphere direction u.
    h_filter: ndarray (W, H, D)
        3-dimensional filter to apply.
    sigma_range: float
        Standard deviation of the Gaussian distribution defining the
        range kernel.

    Returns
    -------
    out_im: ndarray (X, Y, Z)
        Filtered SF image.
    """
    h_w, h_h, h_d = h_filter.shape[:3]
    half_w, half_h, half_d = h_w // 2, h_h // 2, h_d // 2
    out_im = np.zeros_like(image_u)
    image_u = np.pad(image_u, ((half_w, half_w),
                               (half_h, half_h),
                               (half_d, half_d)))

    for ii in range(out_im.shape[0]):
        for jj in range(out_im.shape[1]):
            for kk in range(out_im.shape[2]):
                x = image_u[ii:ii+h_w, jj:jj+h_h, kk:kk+h_d]\
                    - image_u[ii, jj, kk]
                range_filter = _evaluate_gaussian_distribution(x, sigma_range)
                res_filter = range_filter * h_filter

                out_im[ii, jj, kk] += np.sum(image_u[ii:ii+h_w,
                                                     jj:jj+h_h,
                                                     kk:kk+h_d]
                                             * res_filter)
                out_im[ii, jj, kk] /= np.sum(res_filter)

    return out_im


def cosine_filtering(in_sh, sh_order=8, sh_basis='descoteaux07',
                     in_full_basis=False, dot_sharpness=1.0,
                     sphere_str='repulsion724', sigma=1.0):
    """
    Average the SH projected on a sphere using a first-neighbor gaussian
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
    dot_sharpness: float, optional
        Exponent of the dot product. When set to 0.0, directions
        are not weighted by the dot product.
    sphere_str: str, optional
        Name of the sphere used to project SH coefficients to SF.
    sigma: float, optional
        Sigma for the Gaussian.

    Returns
    -------
    out_sh: ndarray (x, y, z, n_coeffs)
        Filtered signal as SH coefficients in full SH basis.
    """
    # Load the sphere used for projection of SH
    sphere = get_sphere(sphere_str)

    # Normalized filter for each sf direction
    weights = _get_weights(sphere, dot_sharpness, sigma)

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

        # Calculate contribution of center voxel
        current_sf = np.dot(in_sh, B[:, sf_i])
        mean_sf[..., sf_i] = w_filter[1, 1, 1] * current_sf

        # Add contributions of neighbors using opposite hemispheres
        current_sf = np.dot(in_sh, neg_B[:, sf_i])
        w_filter[1, 1, 1] = 0.0
        mean_sf[..., sf_i] += correlate(current_sf, w_filter, mode="constant")

    # Convert back to SH coefficients
    _, B_inv = sh_to_sf_matrix(sphere, sh_order=sh_order,
                               basis_type=sh_basis,
                               full_basis=True)

    out_sh = np.array([np.dot(i, B_inv) for i in mean_sf], dtype=in_sh.dtype)
    return out_sh


def _get_weights(sphere, dot_sharpness, sigma):
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
    weights: dictionary
        Vertices weights with respect to voxel directions.
    """
    directions = np.zeros((3, 3, 3, 3))
    for x in range(3):
        for y in range(3):
            for z in range(3):
                directions[x, y, z, 0] = x - 1
                directions[x, y, z, 1] = y - 1
                directions[x, y, z, 2] = z - 1

    non_zero_dir = np.ones((3, 3, 3), dtype=bool)
    non_zero_dir[1, 1, 1] = False

    # normalize dir
    dir_norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions[non_zero_dir] /= dir_norm[non_zero_dir]

    g_weights = np.exp(-dir_norm**2 / (2 * sigma**2))
    d_weights = np.dot(directions, sphere.vertices.T)

    d_weights = np.where(d_weights > 0.0, d_weights**dot_sharpness, 0.0)
    weights = d_weights * g_weights
    weights[1, 1, 1, :] = 1.0

    # Normalize filters so that all sphere directions weights sum to 1
    weights /= weights.reshape((-1, weights.shape[-1])).sum(axis=0)

    return weights
