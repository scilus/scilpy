# -*- coding: utf-8 -*-
import numpy as np
import logging
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from dipy.core.sphere import Sphere
from scipy.ndimage import correlate
from itertools import product as iterprod
from scilpy.gpuparallel.opencl_utils import have_opencl, CLKernel, CLManager


class AsymmetricFilter():
    def __init__(self, sh_order, sh_basis, legacy, full_basis,
                 sphere_str, sigma_spatial, sigma_align,
                 sigma_angle, sigma_range, disable_spatial=False,
                 disable_align=False, disable_angle=False,
                 disable_range=False, device_type='gpu',
                 j_invariance=False):
        self.sh_order = sh_order
        self.legacy = legacy
        self.basis = sh_basis
        self.full_basis = full_basis
        self.sphere = get_sphere(sphere_str)
        self.sigma_spatial = sigma_spatial
        self.sigma_align = sigma_align
        self.sigma_angle = sigma_angle
        self.sigma_range = sigma_range
        self.disable_range = disable_range
        self.disable_angle = disable_angle
        self.device_type = device_type

        if device_type == 'gpu' and not have_opencl:
            raise ValueError('device_type `gpu` requested but pyopencl'
                             ' is not installed.\nPlease install pyopencl to'
                             ' enable GPU acceleration.')

        # won't need this if disable angle
        self.uv_filter = self._build_uv_filter(self.sphere.vertices,
                                               self.sigma_angle)

        # sigma still controls the width of the filter
        self.nx_filter = self._build_nx_filter(self.sphere.vertices, sigma_spatial,
                                               sigma_align, disable_spatial,
                                               disable_align, j_invariance)

        self.B = sh_to_sf_matrix(self.sphere, self.sh_order,
                                 self.basis, self.full_basis,
                                 legacy=self.legacy, return_inv=False)
        _, self.B_inv = sh_to_sf_matrix(self.sphere, self.sh_order,
                                        self.basis, True, legacy=self.legacy,
                                        return_inv=True)

        # initialize gpu
        self.cl_kernel = None
        self.cl_manager = None
        self._prepare_gpu()

    def _prepare_gpu(self):
        self.cl_kernel = CLKernel('filter', 'filtering', 'aodf_filter.cl')

        self.cl_kernel.set_define('WIN_WIDTH', self.nx_filter.shape[0])
        self.cl_kernel.set_define('SIGMA_RANGE',
                                  '{}f'.format(self.sigma_range))
        self.cl_kernel.set_define('N_DIRS', len(self.sphere.vertices))
        self.cl_kernel.set_define('DISABLE_ANGLE', 'true' if self.disable_angle
                                                   else 'false')
        self.cl_kernel.set_define('DISABLE_RANGE', 'true' if self.disable_range
                                                   else 'false')
        self.cl_manager = CLManager(self.cl_kernel, self.device_type)

    def __call__(self, sh_data, patch_size=40):
        uv_weights_offsets =\
            np.append([0.0], np.cumsum(np.count_nonzero(self.uv_filter, axis=-1)))
        v_indices = np.tile(np.arange(self.uv_filter.shape[1]),
                            (self.uv_filter.shape[0], 1))[self.uv_filter > 0.0]
        flat_uv = self.uv_filter[self.uv_filter > 0.0]

        # Prepare GPU buffers
        self.cl_manager.add_input_buffer("sf_data")  # SF data not initialized
        self.cl_manager.add_input_buffer("nx_filter", self.nx_filter)
        self.cl_manager.add_input_buffer("uv_filter", flat_uv)
        self.cl_manager.add_input_buffer("uv_weights_offsets", uv_weights_offsets)
        self.cl_manager.add_input_buffer("v_indices", v_indices)
        self.cl_manager.add_output_buffer("out_sf")  # SF not initialized yet

        win_width = self.nx_filter.shape[0]
        win_hwidth = win_width // 2
        volume_shape = sh_data.shape[:-1]
        padded_volume_shape = tuple(np.asarray(volume_shape) + win_width - 1)

        out_sh = np.zeros(np.append(sh_data.shape[:-1], self.B_inv.shape[-1]))
        # Pad SH data
        sh_data = np.pad(sh_data, ((win_hwidth, win_hwidth),
                                   (win_hwidth, win_hwidth),
                                   (win_hwidth, win_hwidth),
                                   (0, 0)))

        # process in batches
        padded_patch_size = patch_size + self.nx_filter.shape[0] - 1

        n_splits = np.ceil(np.asarray(volume_shape) / float(patch_size))\
            .astype(int)
        splits_prod = iterprod(np.arange(n_splits[0]),
                               np.arange(n_splits[1]),
                               np.arange(n_splits[2]))
        n_splits_prod = np.prod(n_splits)
        for i, split_offset in enumerate(splits_prod):
            logging.info('Patch {}/{}'.format(i+1, n_splits_prod))
            i, j, k = split_offset
            patch_in = np.array(
                [[i * patch_size, min((i*patch_size)+padded_patch_size,
                                      padded_volume_shape[0])],
                 [j * patch_size, min((j*patch_size)+padded_patch_size,
                                      padded_volume_shape[1])],
                 [k * patch_size, min((k*patch_size)+padded_patch_size,
                                      padded_volume_shape[2])]])
            patch_out = np.array(
                [[i * patch_size, min((i+1)*patch_size, volume_shape[0])],
                 [j * patch_size, min((j+1)*patch_size, volume_shape[1])],
                 [k * patch_size, min((k+1)*patch_size, volume_shape[2])]])
            out_shape = tuple(np.append(patch_out[:, 1] - patch_out[:, 0],
                                        len(self.sphere.vertices)))

            sh_patch = sh_data[patch_in[0, 0]:patch_in[0, 1],
                               patch_in[1, 0]:patch_in[1, 1],
                               patch_in[2, 0]:patch_in[2, 1]]

            sf_patch = np.dot(sh_patch, self.B)
            self.cl_manager.update_input_buffer("sf_data", sf_patch)
            self.cl_manager.update_output_buffer("out_sf", out_shape)
            out_sf = self.cl_manager.run(out_shape[:-1])[0]
            out_sh[patch_out[0, 0]:patch_out[0, 1],
                   patch_out[1, 0]:patch_out[1, 1],
                   patch_out[2, 0]:patch_out[2, 1]] = np.dot(out_sf,
                                                             self.B_inv)

        return out_sh

    def _build_uv_filter(directions, sigma_angle):
        dot = directions.dot(directions.T)
        x = np.arccos(np.clip(dot, -1.0, 1.0))
        weights = _evaluate_gaussian_distribution(sigma_angle, x)

        mask = x > (3.0*sigma_angle)
        weights[mask] = 0.0
        weights /= np.sum(weights, axis=-1)
        return weights

    def _build_nx_filter(directions, sigma_spatial, sigma_align,
                        disable_spatial=False, disable_align=False,
                        j_invariance=False):
        directions = directions.astype(np.float32)
        half_width = int(round(3*sigma_spatial))
        filter_shape = (half_width*2+1, half_width*2+1, half_width*2+1)

        grid_directions = _get_window_directions(filter_shape).astype(np.float32)
        distances = np.linalg.norm(grid_directions, axis=-1)
        grid_directions[distances > 0] = grid_directions[distances > 0]\
                                        / distances[distances > 0][..., None]

        if disable_spatial:
            w_spatial = np.ones(filter_shape)
        else:
            w_spatial = _evaluate_gaussian_distribution(sigma_spatial, distances)

        cos_theta = np.clip(grid_directions.dot(directions.T), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        theta[half_width, half_width, half_width] = 0.0

        if disable_align:
            w_align = np.ones(np.append(filter_shape, (len(directions),)))
        else:
            w_align = _evaluate_gaussian_distribution(sigma_align, theta)

        w = w_spatial[..., None] * w_align

        if j_invariance:
            w[half_width, half_width, half_width] = 0.0

        # normalize
        w /= np.sum(w, axis=(0,1,2), keepdims=True)

        return w


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
