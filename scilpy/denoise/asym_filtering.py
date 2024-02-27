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
    """
    Unified asymmetric filtering as described in [1].

    Parameters
    ----------
    sh_order: int
        Maximum order of spherical harmonics (SH) basis.
    sh_basis: str
        SH basis definition used for input and output SH image.
        One of 'descoteaux07' or 'tournier07'.
    legacy: bool
        Whether the legacy SH basis definition should be used.
    full_basis: bool
        Whether the input SH basis is full or not.
    sphere_str: str
        Name of the DIPY sphere to use for SH to SF projection.
    sigma_spatial: float
        Standard deviation of spatial filter. Also controls the
        filtering window size (6*sigma_spatial + 1).
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
    j_invariance: bool, optional
        Assign a weight of 0 to the center voxel of the filter.
    device_type: string, optional
        Device on which the code should run. Choices are cpu or gpu.
    use_opencl: bool, optional
        Use OpenCL for software acceleration.

    References
    ----------
    [1] Poirier and Descoteaux, 2024, "A Unified Filtering Method for
        Estimating Asymmetric Orientation Distribution Functions",
        Neuroimage, https://doi.org/10.1016/j.neuroimage.2024.120516
    """
    def __init__(self, sh_order, sh_basis, legacy, full_basis, sphere_str,
                 sigma_spatial=1.0, sigma_align=0.8, sigma_angle=None,
                 rel_sigma_range=0.2, disable_spatial=False,
                 j_invariance=False, device_type='gpu',
                 use_opencl=True):
        self.sh_order = sh_order
        self.legacy = legacy
        self.basis = sh_basis
        self.full_basis = full_basis
        self.sphere = get_sphere(sphere_str)
        self.rel_sigma_range = rel_sigma_range
        self.disable_range = rel_sigma_range is None
        self.disable_angle = sigma_angle is None
        self.disable_spatial = disable_spatial
        self.disable_align = sigma_align is None
        self.device_type = device_type
        self.use_opencl = use_opencl
        self.j_invariance = j_invariance

        if device_type not in ['cpu', 'gpu']:
            raise ValueError('Invalid device type {}. Must be cpu or gpu'
                             .format(device_type))
        if use_opencl and not have_opencl:
            raise ValueError('pyopencl is not installed. Please install before'
                             ' using option use_opencl=True.')
        if device_type == 'gpu' and not use_opencl:
            raise ValueError('Option use_opencl must be enabled '
                             'to use device \'gpu\'.')

        # build filters
        self.uv_filter = self._build_uv_filter(sigma_angle)
        self.nx_filter = self._build_nx_filter(sigma_spatial, sigma_align)

        self.B = sh_to_sf_matrix(self.sphere, self.sh_order,
                                 self.basis, self.full_basis,
                                 legacy=self.legacy, return_inv=False)
        _, self.B_inv = sh_to_sf_matrix(self.sphere, self.sh_order,
                                        self.basis, True, legacy=self.legacy,
                                        return_inv=True)

        if self.use_opencl:
            # initialize opencl
            self.cl_kernel = None
            self.cl_manager = None

    def _prepare_opencl(self, sigma_range):
        """
        Instantiate OpenCL context manager and compile OpenCL program.

        Parameters
        ----------
        sigma_range: float
            Value for sigma_range. Will be cast to float. Must be provided even
            when range filtering is disabled for the OpenCL program to compile.
        """
        self.cl_kernel = CLKernel('filter', 'denoise', 'aodf_filter.cl')

        self.cl_kernel.set_define('WIN_WIDTH', self.nx_filter.shape[0])
        self.cl_kernel.set_define('SIGMA_RANGE',
                                  '{}f'.format(sigma_range))
        self.cl_kernel.set_define('N_DIRS', len(self.sphere.vertices))
        self.cl_kernel.set_define('DISABLE_ANGLE', 'true' if self.disable_angle
                                                   else 'false')
        self.cl_kernel.set_define('DISABLE_RANGE', 'true' if self.disable_range
                                                   else 'false')
        self.cl_manager = CLManager(self.cl_kernel, self.device_type)

    def _build_uv_filter(self, sigma_angle):
        """
        Build the angle filter.

        Parameters
        ----------
        sigma_angle: float
            Standard deviation of filter. Values at distances greater than
            sigma_angle are clipped to 0 to reduce computation time.
            Ignored when self.disable_angle is True.

        Returns
        -------
        weights: ndarray
            Angle filter of shape (N_dirs, N_dirs).
        """
        directions = self.sphere.vertices
        if not self.disable_angle:
            dot = directions.dot(directions.T)
            x = np.arccos(np.clip(dot, -1.0, 1.0))
            weights = _evaluate_gaussian_distribution(x, sigma_angle)
            mask = x > (3.0*sigma_angle)
            weights[mask] = 0.0
            weights /= np.sum(weights, axis=-1)
        else:
            weights = np.eye(len(directions))
        return weights

    def _build_nx_filter(self, sigma_spatial, sigma_align):
        """
        Build the combined spatial and alignment filter.

        Parameters
        ----------
        sigma_spatial: float
            Standard deviation of spatial filter. Also controls the
            filtering window size (6*sigma_spatial + 1).
        sigma_align: float
            Standard deviation of the alignment filter. Ignored when
            self.disable_align is True.

        Returns
        -------
        weights: ndarray
            Combined spatial + alignment filter of shape (W, H, D, N) where
            N is the number of sphere directions.
        """
        directions = self.sphere.vertices.astype(np.float32)
        half_width = int(round(3*sigma_spatial))
        filter_shape = (half_width*2+1, half_width*2+1, half_width*2+1)

        grid_directions =\
            _get_window_directions(filter_shape).astype(np.float32)
        distances = np.linalg.norm(grid_directions, axis=-1)
        grid_directions[distances > 0] =\
            grid_directions[distances > 0] /\
            distances[distances > 0][..., None]

        if self.disable_spatial:
            w_spatial = np.ones(filter_shape)
        else:
            w_spatial = _evaluate_gaussian_distribution(distances,
                                                        sigma_spatial)

        if self.disable_align:
            w_align = np.ones(np.append(filter_shape, (len(directions),)))
        else:
            cos_theta = np.clip(grid_directions.dot(directions.T), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            theta[half_width, half_width, half_width] = 0.0
            w_align = _evaluate_gaussian_distribution(theta, sigma_align)

        w = w_spatial[..., None] * w_align

        if self.j_invariance:
            w[half_width, half_width, half_width] = 0.0

        # normalize
        w /= np.sum(w, axis=(0, 1, 2), keepdims=True)

        return w

    def _get_sf_range(self, sh_data):
        """
        Get the range of SF amplitudes for input `sh_data`.

        Parameters
        ----------
        sh_data: ndarray
            Spherical harmonics coefficients image.

        Returns
        -------
        sf_range: float
            Range of SF amplitudes.
        """
        sf_max = 0.0
        sf_min = np.inf
        for sph_id in range(len(self.sphere.vertices)):
            sf = np.dot(sh_data, self.B[:, sph_id])
            sf[sf < 0.0] = 0.0
            sf_max = max(sf_max, np.max(sf))
            sf_min = min(sf_min, np.min(sf))
        return sf_max - sf_min

    def _call_opencl(self, sh_data, patch_size):
        """
        Run filtering using opencl.

        Parameters
        ----------
        sh_data: ndarray
            Input SH volume.
        patch_size: int
            Data is processed in patches of
            patch_size x patch_size x patch_size.

        Returns
        -------
        out_sh: ndarray
            Filtered output as SH coefficients.
        """
        uv_weights_offsets =\
            np.append([0.0], np.cumsum(np.count_nonzero(self.uv_filter,
                                                        axis=-1)))
        v_indices = np.tile(np.arange(self.uv_filter.shape[1]),
                            (self.uv_filter.shape[0], 1))[self.uv_filter > 0.0]
        flat_uv = self.uv_filter[self.uv_filter > 0.0]

        # Prepare GPU buffers
        self.cl_manager.add_input_buffer("sf_data")  # SF data not initialized
        self.cl_manager.add_input_buffer("nx_filter", self.nx_filter)
        self.cl_manager.add_input_buffer("uv_filter", flat_uv)
        self.cl_manager.add_input_buffer("uv_weights_offsets",
                                         uv_weights_offsets)
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

    def _call_purepython(self, sh_data, sigma_range):
        """
        Run filtering using pure python implementation.

        Parameters
        ----------
        sh_data: ndarray
            Input SH data.
        sigma_range: float
            Standard deviation of range filter. Ignored when
            self.disable_range is True.

        Returns
        -------
        out_sh: ndarray
            Filtered output as SH coefficients.
        """
        nb_sf = len(self.sphere.vertices)
        mean_sf = np.zeros(sh_data.shape[:-1] + (nb_sf,))

        # Apply filter to each sphere vertice
        for u_sph_id in range(nb_sf):
            if u_sph_id % 20 == 0:
                logging.info('Processing direction: {}/{}'
                             .format(u_sph_id, nb_sf))
            mean_sf[..., u_sph_id] = self._correlate(sh_data, sigma_range,
                                                     u_sph_id)

        out_sh = np.array([np.dot(i, self.B_inv) for i in mean_sf],
                          dtype=sh_data.dtype)
        return out_sh

    def _correlate(self, sh_data, sigma_range, u_index):
        """
        Apply the filters to the SH image for the sphere direction
        described by `u_index`.

        Parameters
        ----------
        sh_data: ndarray
            Input SH coefficients.
        sigma_range: float
            Standard deviation of range filter. Ignored when
            self.disable_range is True.
        u_index: int
            Index of the current sphere direction to process.

        Returns
        -------
        out_sf: ndarray
            Output SF amplitudes along the direction described by `u_index`.
        """
        v_indices = np.flatnonzero(self.uv_filter[u_index])
        nx_filter = self.nx_filter[..., u_index]
        h_w, h_h, h_d = nx_filter.shape[:3]
        half_w, half_h, half_d = h_w // 2, h_h // 2, h_d // 2
        out_sf = np.zeros(sh_data.shape[:3])
        sh_data = np.pad(sh_data, ((half_w, half_w),
                                   (half_h, half_h),
                                   (half_d, half_d),
                                   (0, 0)))

        sf_u = np.dot(sh_data, self.B[:, u_index])
        sf_v = np.dot(sh_data, self.B[:, v_indices])
        uv_filter = self.uv_filter[u_index, v_indices]

        _get_range = _evaluate_gaussian_distribution\
            if not self.disable_range else lambda x, _: np.ones_like(x)

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

    def __call__(self, sh_data, patch_size=40):
        """
        Run filtering.

        Parameters
        ----------
        sh_data: ndarray
            Input SH coefficients.
        patch_size: int, optional
            Data is processed in patches of
            patch_size x patch_size x patch_size.

        Returns
        -------
        out_sh: ndarray
            Filtered output as SH coefficients.
        """
        sigma_range = 0.0
        if not self.disable_range:
            sigma_range = self.rel_sigma_range * self._get_sf_range(sh_data)
        if self.use_opencl:
            self._prepare_opencl(sigma_range)
            return self._call_opencl(sh_data, patch_size)
        else:
            return self._call_purepython(sh_data, sigma_range)


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
    weights = _get_cosine_weights(sphere, dot_sharpness, sigma)

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
