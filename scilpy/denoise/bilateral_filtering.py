# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing
import itertools
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from scilpy.gpuparallel.opencl_utils import (have_opencl, CLKernel, CLManager)


def angle_aware_bilateral_filtering(in_sh, sh_order=8,
                                    sh_basis='descoteaux07',
                                    in_full_basis=False,
                                    sphere_str='repulsion724',
                                    sigma_spatial=1.0,
                                    sigma_angular=1.0,
                                    sigma_range=0.5,
                                    use_gpu=True,
                                    nbr_processes=1):
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
    nbr_processes: int, optional
        Number of processes to use.

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
                                                   sigma_angular, sigma_range,
                                                   nbr_processes)


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
                                        sigma_range=0.5,
                                        nbr_processes=1):
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
    nbr_processes: int, optional
        Number of processes to use.

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

    if nbr_processes > 1:
        # Apply filter to each sphere vertice in parallel
        pool = multiprocessing.Pool(nbr_processes)

        # divide the sphere directions among the processes
        base_chunk_size = int(nb_sf / nbr_processes + 0.5)
        first_ids = np.arange(0, nb_sf, base_chunk_size)
        residuals = nb_sf - first_ids
        chunk_sizes = np.where(residuals < base_chunk_size,
                               residuals, base_chunk_size)
        res = pool.map(_process_subset_directions,
                       zip(itertools.repeat(weights),
                           itertools.repeat(in_sh),
                           first_ids,
                           chunk_sizes,
                           itertools.repeat(B),
                           itertools.repeat(sigma_range)))
        pool.close()
        pool.join()

        # Patch chunks together.
        mean_sf = np.concatenate(res, axis=-1)
    else:
        args = [weights, in_sh, 0, nb_sf,
                B, sigma_range]
        mean_sf = _process_subset_directions(args)

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


def _process_subset_directions(args):
    """
    Filter a subset of all sphere directions.

    Parameters
    ----------
    args: List
        args[0]: weights, ndarray
            Filter weights per direction.
        args[1]: in_sh, ndarray
            Input SH coefficients array.
        args[2]: first_dir_id, int
            ID of first sphere direction.
        args[3]: chunk_size, int
            Number of sphere directions in chunk.
        args[4]: B, ndarray
            SH to SF matrix for current sphere directions.
        args[5]: neg_B, ndarray
            SH to SF matrix for opposite sphere directions.
        args[6]: sigma_range, int
            Sigma of the Gaussian use for range filtering.

    Returns
    -------
    out_sf: ndarray
        SF array for subset directions.
    """
    weights = args[0]
    in_sh = args[1]
    first_dir_id = args[2]
    chunk_size = args[3]
    B = args[4]
    sigma_range = args[5]

    out_sf = np.zeros(in_sh.shape[:-1] + (chunk_size,))
    # Apply filter to each sphere vertice
    for offset_i in range(chunk_size):
        sph_id = first_dir_id + offset_i
        w_filter = weights[..., sph_id]

        # Generate 1-channel images for directions u and -u
        current_sf = np.dot(in_sh, B[:, sph_id])
        out_sf[..., offset_i] = _correlate_spatial(current_sf,
                                                   w_filter,
                                                   sigma_range)
    return out_sf


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
