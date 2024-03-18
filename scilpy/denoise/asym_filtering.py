# -*- coding: utf-8 -*-
import numpy as np
import logging
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from dipy.core.sphere import Sphere
from scipy.ndimage import correlate
from itertools import product as iterprod
from scilpy.gpuparallel.opencl_utils import have_opencl, CLKernel, CLManager


def unified_filtering(sh_data, sh_order, sh_basis, is_legacy, full_basis,
                      sphere_str, sigma_spatial=1.0, sigma_align=0.8,
                      sigma_angle=None, rel_sigma_range=0.2,
                      win_hwidth=None, exclude_center=False,
                      device_type='gpu', use_opencl=True, patch_size=40):
    """
    Unified asymmetric filtering as described in [1].

    Parameters
    ----------
    sh_data: ndarray
        SH coefficients image.
    sh_order: int
        Maximum order of spherical harmonics (SH) basis.
    sh_basis: str
        SH basis definition used for input and output SH image.
        One of 'descoteaux07' or 'tournier07'.
    is_legacy: bool
        Whether the legacy SH basis definition should be used.
    full_basis: bool
        Whether the input SH basis is full or not.
    sphere_str: str
        Name of the DIPY sphere to use for SH to SF projection.
    sigma_spatial: float or None
        Standard deviation of spatial filter. Can be None to replace
        by mean filter, in what case win_hwidth must be given.
    sigma_align: float or None
        Standard deviation of alignment filter. `None` disables
        alignment filtering.
    sigma_angle: float or None
        Standard deviation of the angle filter. `None` disables
        angle filtering.
    rel_sigma_range: float or None
        Standard deviation of the range filter, relative to the
        range of SF amplitudes. `None` disables range filtering.
    disable_spatial: bool, optional
        Replace gaussian filter by a mean filter for spatial filter.
        The value from `sigma_spatial` is still used for setting the
        size of the filtering window.
    win_hwidth: int, optional
        Half-width of the filtering window. When None, the
        filtering window half-width is given by (6*sigma_spatial + 1).
    exclude_center: bool, optional
        Assign a weight of 0 to the center voxel of the filter.
    device_type: string, optional
        Device on which the code should run. Choices are cpu or gpu.
    use_opencl: bool, optional
        Use OpenCL for software acceleration.
    patch_size: int, optional
        Patch size for OpenCL execution.

    References
    ----------
    [1] Poirier and Descoteaux, 2024, "A Unified Filtering Method for
        Estimating Asymmetric Orientation Distribution Functions",
        Neuroimage, https://doi.org/10.1016/j.neuroimage.2024.120516
    """
    if sigma_spatial is None and win_hwidth is None:
        raise ValueError('sigma_spatial and win_hwidth cannot both be None')
    if device_type not in ['cpu', 'gpu']:
        raise ValueError('Invalid device type {}. Must be cpu or gpu'
                         .format(device_type))
    if use_opencl and not have_opencl:
        raise ValueError('pyopencl is not installed. Please install before'
                         ' using option use_opencl=True.')
    if device_type == 'gpu' and not use_opencl:
        raise ValueError('Option use_opencl must be enabled '
                         'to use device \'gpu\'.')

    sphere = get_sphere(sphere_str)

    if sigma_spatial is not None:
        if sigma_spatial <= 0.0:
            raise ValueError('sigma_spatial cannot be <= 0.')
        # calculate half-width from sigma_spatial
        half_width = int(round(3*sigma_spatial))
    if sigma_align is not None:
        if sigma_align <= 0.0:
            raise ValueError('sigma_align cannot be <= 0.')
    if sigma_angle is not None:
        if sigma_angle <= 0.0:
            raise ValueError('sigma_align cannot be <= 0.')

    # overwrite half-width if win_hwidth is supplied
    if win_hwidth is not None:
        half_width = win_hwidth

    # filter shape computed from half_width
    filter_shape = (half_width*2+1, half_width*2+1, half_width*2+1)

    # build filters
    uv_filter = _unified_filter_build_uv(sigma_angle, sphere)
    nx_filter = _unified_filter_build_nx(filter_shape, sigma_spatial,
                                         sigma_align, sphere, exclude_center)

    B = sh_to_sf_matrix(sphere, sh_order, sh_basis, full_basis,
                        legacy=is_legacy, return_inv=False)
    _, B_inv = sh_to_sf_matrix(sphere, sh_order, sh_basis, True,
                               legacy=is_legacy, return_inv=True)

    # compute "real" sigma_range scaled by sf amplitudes
    # if rel_sigma_range is supplied
    sigma_range = None
    if rel_sigma_range is not None:
        if rel_sigma_range <= 0.0:
            raise ValueError('sigma_rangel cannot be <= 0.')
        sigma_range = rel_sigma_range * _get_sf_range(sh_data, B)

    if use_opencl:
        # initialize opencl
        cl_manager = _unified_filter_prepare_opencl(sigma_range, sigma_angle,
                                                    filter_shape[0], sphere,
                                                    device_type)

        return _unified_filter_call_opencl(sh_data, nx_filter, uv_filter,
                                           cl_manager, B, B_inv, sphere,
                                           patch_size)
    else:
        return _unified_filter_call_python(sh_data, nx_filter, uv_filter,
                                           sigma_range, B, B_inv, sphere)


def _unified_filter_prepare_opencl(sigma_range, sigma_angle, window_width,
                                   sphere, device_type):
    """
    Instantiate OpenCL context manager and compile OpenCL program.

    Parameters
    ----------
    sigma_range: float or None
        Value for sigma_range.
    sigma_angle: float or None
        Value for sigma_angle.
    window_width: int
        Width of filtering window.
    sphere: DIPY sphere
        Sphere used for SH to SF projection.
    device_type: string
        Device to be used by OpenCL. Either 'cpu' or 'gpu'.

    Returns
    -------
    cl_manager: CLManager
        OpenCL manager object.
    """
    disable_range = sigma_range is None
    disable_angle = sigma_angle is None
    if sigma_range is None:
        sigma_range = 0.0  # placeholder value for sigma_range

    cl_kernel = CLKernel('filter', 'denoise', 'aodf_filter.cl')
    cl_kernel.set_define('WIN_WIDTH', window_width)
    cl_kernel.set_define('SIGMA_RANGE', '{}f'.format(sigma_range))
    cl_kernel.set_define('N_DIRS', len(sphere.vertices))
    cl_kernel.set_define('DISABLE_ANGLE', 'true' if disable_angle else 'false')
    cl_kernel.set_define('DISABLE_RANGE', 'true' if disable_range else 'false')

    return CLManager(cl_kernel, device_type)


def _unified_filter_build_uv(sigma_angle, sphere):
    """
    Build the angle filter, weighted on angle between current direction u
    and neighbour direction v.

    Parameters
    ----------
    sigma_angle: float
        Standard deviation of filter. Values at distances greater than
        sigma_angle are clipped to 0 to reduce computation time.
    sphere: DIPY sphere
        Sphere used for sampling the SF.

    Returns
    -------
    weights: ndarray
        Angle filter of shape (N_dirs, N_dirs).
    """
    directions = sphere.vertices
    if sigma_angle is not None:
        dot = directions.dot(directions.T)
        x = np.arccos(np.clip(dot, -1.0, 1.0))
        weights = _evaluate_gaussian_distribution(x, sigma_angle)
        mask = x > (3.0*sigma_angle)
        weights[mask] = 0.0
        weights /= np.sum(weights, axis=-1)
    else:
        weights = np.eye(len(directions))
    return weights


def _unified_filter_build_nx(filter_shape, sigma_spatial, sigma_align,
                             sphere, exclude_center):
    """
    Build the combined spatial and alignment filter.

    Parameters
    ----------
    filter_shape: tuple
        Dimensions of filtering window.
    sigma_spatial: float or None
        Standard deviation of spatial filter. None disables Gaussian
        weighting for spatial filtering.
    sigma_align: float or None
        Standard deviation of the alignment filter. None disables Gaussian
        weighting for alignment filtering.
    sphere: DIPY sphere
        Sphere for SH to SF projection.
    exclude_center: bool
        Whether the center voxel is included in the neighbourhood.

    Returns
    -------
    weights: ndarray
        Combined spatial + alignment filter of shape (W, H, D, N) where
        N is the number of sphere directions.
    """
    directions = sphere.vertices.astype(np.float32)

    grid_directions = _get_window_directions(filter_shape).astype(np.float32)
    distances = np.linalg.norm(grid_directions, axis=-1)
    grid_directions[distances > 0] = grid_directions[distances > 0] /\
        distances[distances > 0][..., None]

    if sigma_spatial is None:
        w_spatial = np.ones(filter_shape)
    else:
        w_spatial = _evaluate_gaussian_distribution(distances, sigma_spatial)

    if sigma_align is None:
        w_align = np.ones(np.append(filter_shape, (len(directions),)))
    else:
        cos_theta = np.clip(grid_directions.dot(directions.T), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        theta[filter_shape[0] // 2,
              filter_shape[1] // 2,
              filter_shape[2] // 2] = 0.0
        w_align = _evaluate_gaussian_distribution(theta, sigma_align)

    # resulting filter
    w = w_spatial[..., None] * w_align

    if exclude_center:
        w[filter_shape[0] // 2,
          filter_shape[1] // 2,
          filter_shape[2] // 2] = 0.0

    # normalize and return
    w /= np.sum(w, axis=(0, 1, 2), keepdims=True)
    return w


def _get_sf_range(sh_data, B_mat):
    """
    Get the range of SF amplitudes for input `sh_data`.

    Parameters
    ----------
    sh_data: ndarray
        Spherical harmonics coefficients image.
    B_mat: ndarray
        SH to SF projection matrix.

    Returns
    -------
    sf_range: float
        Range of SF amplitudes.
    """
    sf = np.array([np.dot(i, B_mat) for i in sh_data],
                  dtype=sh_data.dtype)
    sf[sf < 0.0] = 0.0
    sf_max = np.max(sf)
    sf_min = np.min(sf)
    return sf_max - sf_min


def _unified_filter_call_opencl(sh_data, nx_filter, uv_filter, cl_manager,
                                B, B_inv, sphere, patch_size=40):
    """
    Run unified filtering for asymmetric ODFs using OpenCL.

    Parameters
    ----------
    sh_data: ndarray
        Input SH volume.
    nx_filter: ndarray
        Combined spatial and alignment filter.
    uv_filter: ndarray
        Angle filter.
    cl_manager: CLManager
        A CLManager instance.
    B: ndarray
        SH to SF projection matrix.
    B_inv: ndarray
        SF to SH projection matrix.
    sphere: DIPY sphere
        Sphere for SH to SF projection.
    patch_size: int
        Data is processed in patches of
        patch_size x patch_size x patch_size.

    Returns
    -------
    out_sh: ndarray
        Filtered output as SH coefficients.
    """
    uv_weights_offsets =\
        np.append([0.0], np.cumsum(np.count_nonzero(uv_filter,
                                                    axis=-1)))
    v_indices = np.tile(np.arange(uv_filter.shape[1]),
                        (uv_filter.shape[0], 1))[uv_filter > 0.0]
    flat_uv = uv_filter[uv_filter > 0.0]

    # Prepare GPU buffers
    cl_manager.add_input_buffer("sf_data")  # SF data not initialized
    cl_manager.add_input_buffer("nx_filter", nx_filter)
    cl_manager.add_input_buffer("uv_filter", flat_uv)
    cl_manager.add_input_buffer("uv_weights_offsets", uv_weights_offsets)
    cl_manager.add_input_buffer("v_indices", v_indices)
    cl_manager.add_output_buffer("out_sf")  # SF not initialized yet

    win_width = nx_filter.shape[0]
    win_hwidth = win_width // 2
    volume_shape = sh_data.shape[:-1]
    padded_volume_shape = tuple(np.asarray(volume_shape) + win_width - 1)

    out_sh = np.zeros(np.append(sh_data.shape[:-1], B_inv.shape[-1]))
    # Pad SH data
    sh_data = np.pad(sh_data, ((win_hwidth, win_hwidth),
                               (win_hwidth, win_hwidth),
                               (win_hwidth, win_hwidth),
                               (0, 0)))

    # process in batches
    padded_patch_size = patch_size + nx_filter.shape[0] - 1

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
                                    len(sphere.vertices)))

        sh_patch = sh_data[patch_in[0, 0]:patch_in[0, 1],
                           patch_in[1, 0]:patch_in[1, 1],
                           patch_in[2, 0]:patch_in[2, 1]]

        sf_patch = np.dot(sh_patch, B)
        cl_manager.update_input_buffer("sf_data", sf_patch)
        cl_manager.update_output_buffer("out_sf", out_shape)
        out_sf = cl_manager.run(out_shape[:-1])[0]
        out_sh[patch_out[0, 0]:patch_out[0, 1],
               patch_out[1, 0]:patch_out[1, 1],
               patch_out[2, 0]:patch_out[2, 1]] = np.dot(out_sf, B_inv)

    return out_sh


def _unified_filter_call_python(sh_data, nx_filter, uv_filter, sigma_range,
                                B_mat, B_inv, sphere):
    """
    Run filtering using pure python implementation.

    Parameters
    ----------
    sh_data: ndarray
        Input SH data.
    nx_filter: ndarray
        Combined spatial and alignment filter.
    uv_filter: ndarray
        Angle filter.
    sigma_range: float or None
        Standard deviation of range filter. None disables range filtering.
    B_mat: ndarray
        SH to SF projection matrix.
    B_inv: ndarray
        SF to SH projection matrix.
    sphere: DIPY sphere
        Sphere for SH to SF projection.

    Returns
    -------
    out_sh: ndarray
        Filtered output as SH coefficients.
    """
    nb_sf = len(sphere.vertices)
    mean_sf = np.zeros(sh_data.shape[:-1] + (nb_sf,))

    # Apply filter to each sphere vertice
    for u_sph_id in range(nb_sf):
        if u_sph_id % 20 == 0:
            logging.info('Processing direction: {}/{}'
                         .format(u_sph_id, nb_sf))
        mean_sf[..., u_sph_id] = _correlate(sh_data, nx_filter, uv_filter,
                                            sigma_range, u_sph_id, B_mat)

    out_sh = np.array([np.dot(i, B_inv) for i in mean_sf],
                      dtype=sh_data.dtype)
    return out_sh


def _correlate(sh_data, nx_filter, uv_filter, sigma_range, u_index, B_mat):
    """
    Apply the filters to the SH image for the sphere direction
    described by `u_index`.

    Parameters
    ----------
    sh_data: ndarray
        Input SH coefficients.
    nx_filter: ndarray
        Combined spatial and alignment filter.
    uv_filter: ndarray
        Angle filter.
    sigma_range: float or None
        Standard deviation of range filter. None disables range filtering.
    u_index: int
        Index of the current sphere direction to process.
    B_mat: ndarray
        SH to SF projection matrix.

    Returns
    -------
    out_sf: ndarray
        Output SF amplitudes along the direction described by `u_index`.
    """
    v_indices = np.flatnonzero(uv_filter[u_index])
    nx_filter = nx_filter[..., u_index]
    h_w, h_h, h_d = nx_filter.shape[:3]
    half_w, half_h, half_d = h_w // 2, h_h // 2, h_d // 2
    out_sf = np.zeros(sh_data.shape[:3])
    sh_data = np.pad(sh_data, ((half_w, half_w),
                               (half_h, half_h),
                               (half_d, half_d),
                               (0, 0)))

    sf_u = np.dot(sh_data, B_mat[:, u_index])
    sf_v = np.dot(sh_data, B_mat[:, v_indices])
    uv_filter = uv_filter[u_index, v_indices]

    _get_range = _evaluate_gaussian_distribution\
        if sigma_range is not None else lambda x, _: np.ones_like(x)

    for ii in range(out_sf.shape[0]):
        for jj in range(out_sf.shape[1]):
            for kk in range(out_sf.shape[2]):
                a = sf_v[ii:ii+h_w, jj:jj+h_h, kk:kk+h_d]
                b = sf_u[ii + half_w, jj + half_h, kk + half_d]
                x_range = a - b
                range_filter = _get_range(x_range, sigma_range)

                # the resulting filter for the current voxel and v_index
                res_filter = range_filter * nx_filter[..., None]
                res_filter =\
                    res_filter * np.reshape(uv_filter,
                                            (1, 1, 1, len(uv_filter)))
                out_sf[ii, jj, kk] = np.sum(
                    sf_v[ii:ii+h_w, jj:jj+h_h, kk:kk+h_d] * res_filter)
                out_sf[ii, jj, kk] /= np.sum(res_filter)

    return out_sf


def cosine_filtering(in_sh, sh_order=8, sh_basis='descoteaux07',
                     in_full_basis=False, is_legacy=True, dot_sharpness=1.0,
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
    is_legacy : bool, optional
        Whether or not the SH basis is in its legacy form.
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
    weights = _get_cosine_weights(sphere, dot_sharpness, sigma)

    nb_sf = len(sphere.vertices)
    mean_sf = np.zeros(np.append(in_sh.shape[:-1], nb_sf))
    B = sh_to_sf_matrix(sphere, sh_order_max=sh_order, basis_type=sh_basis,
                        return_inv=False, full_basis=in_full_basis,
                        legacy=is_legacy)

    # We want a B matrix to project on an inverse sphere to have the sf on
    # the opposite hemisphere for a given vertice
    neg_B = sh_to_sf_matrix(Sphere(xyz=-sphere.vertices), sh_order_max=sh_order,
                            basis_type=sh_basis, return_inv=False,
                            full_basis=in_full_basis, legacy=is_legacy)

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
    _, B_inv = sh_to_sf_matrix(sphere, sh_order_max=sh_order,
                               basis_type=sh_basis,
                               full_basis=True,
                               legacy=is_legacy)

    out_sh = np.array([np.dot(i, B_inv) for i in mean_sf], dtype=in_sh.dtype)
    return out_sh


def _get_cosine_weights(sphere, dot_sharpness, sigma):
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
