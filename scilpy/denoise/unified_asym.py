# -*- coding: utf-8 -*-
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
import numpy as np
from scilpy.gpuparallel.opencl_utils import CLManager, CLKernel
from itertools import product as iterprod
import logging


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

        # won't need this if disable angle
        self.uv_filter = _build_uv_filter(self.sphere.vertices,
                                          self.sigma_angle)

        # sigma still controls the width of the filter
        self.nx_filter = _build_nx_filter(self.sphere.vertices, sigma_spatial,
                                          sigma_align, disable_spatial,
                                          disable_align, j_invariance)
        logging.info('Filter shape: {}'.format(self.nx_filter.shape[:-1]))

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
    weights = _evaluate_gaussian(sigma_angle, x)

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
        w_spatial = _evaluate_gaussian(sigma_spatial, distances)

    cos_theta = np.clip(grid_directions.dot(directions.T), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta[half_width, half_width, half_width] = 0.0

    if disable_align:
        w_align = np.ones(np.append(filter_shape, (len(directions),)))
    else:
        w_align = _evaluate_gaussian(sigma_align, theta)

    w = w_spatial[..., None] * w_align

    if j_invariance:
        w[half_width, half_width, half_width] = 0.0

    # normalize
    w /= np.sum(w, axis=(0,1,2), keepdims=True)

    return w


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


def _evaluate_gaussian(sigma, x):
    # gaussian is not normalized
    return np.exp(-x**2/(2*sigma**2))
