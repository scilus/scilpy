# -*- coding: utf-8 -*-
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
import numpy as np
from numba import njit
from scilpy.gpuparallel.opencl_utils import CLManager, CLKernel
from itertools import product as iterprod
import logging


class AsymmetricFilter():
    def __init__(self, sh_order, sh_basis, legacy, full_basis,
                 sphere_str, sigma_spatial, sigma_align,
                 sigma_angle, sigma_range, exclude_self=False,
                 disable_spatial=False, disable_align=False,
                 disable_angle=False, disable_range=False):
        self.sh_order = sh_order
        self.legacy = legacy
        self.basis = sh_basis
        self.full_basis = full_basis
        self.sphere = get_sphere(sphere_str)
        self.sigma_spatial = sigma_spatial
        self.sigma_align = sigma_align
        self.sigma_angle = sigma_angle
        self.sigma_range = sigma_range
        self.exclude_self = exclude_self
        self.disable_range = disable_range
        self.disable_angle = disable_angle

        # won't need this if disable range
        self.uv_filter = _build_uv_filter(self.sphere.vertices,
                                          self.sigma_angle)
        # sigma still controls the width of the filter
        self.nx_filter = _build_nx_filter(self.sphere.vertices, sigma_spatial,
                                          sigma_align, disable_spatial,
                                          disable_align)
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
        self.cl_kernel = CLKernel('filter', 'denoise', 'aodf_filter.cl')

        self.cl_kernel.set_define('WIN_WIDTH', self.nx_filter.shape[0])
        self.cl_kernel.set_define('SIGMA_RANGE',
                                  '{}f'.format(self.sigma_range))
        self.cl_kernel.set_define('N_DIRS', len(self.sphere.vertices))
        self.cl_kernel.set_define('EXCLUDE_SELF', 'true' if self.exclude_self
                                                  else 'false')
        self.cl_kernel.set_define('DISABLE_ANGLE', 'true' if self.disable_angle
                                                   else 'false')
        self.cl_kernel.set_define('DISABLE_RANGE', 'true' if self.disable_range
                                                   else 'false')
        self.cl_manager = CLManager(self.cl_kernel, 3, 1)

    def __call__(self, sh_data, patch_size=40):
        # Fill const GPU buffers
        self.cl_manager.add_input_buffer(1, self.nx_filter)
        self.cl_manager.add_input_buffer(2, self.uv_filter)

        win_width = self.nx_filter.shape[0]
        win_hwidth = win_width // 2
        volume_shape = sh_data.shape[:-1]
        sf_shape = tuple(np.append(volume_shape, len(self.sphere.vertices)))
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
            # print(patch_in, sh_patch.shape)
            sf_patch = np.dot(sh_patch, self.B)
            self.cl_manager.add_input_buffer(0, sf_patch)
            self.cl_manager.add_output_buffer(0, out_shape)
            out_sf = self.cl_manager.run(out_shape[:-1])[0]
            out_sh[patch_out[0, 0]:patch_out[0, 1],
                   patch_out[1, 0]:patch_out[1, 1],
                   patch_out[2, 0]:patch_out[2, 1]] = np.dot(out_sf,
                                                             self.B_inv)

        return out_sh


@njit(cache=True)
def _build_uv_filter(directions, sigma_angle):
    directions = np.ascontiguousarray(directions.astype(np.float32))
    uv_weights = np.zeros((len(directions), len(directions)), dtype=np.float32)

    # 1. precompute weights on angle
    # c'est toujours les mêmes peu importe le voxel en question
    for u_i, u in enumerate(directions):
        uvec = np.reshape(np.ascontiguousarray(u), (1, 3))
        weights = np.arccos(np.clip(np.dot(uvec, directions.T), -1.0, 1.0))
        weights = _evaluate_gaussian(sigma_angle, weights)
        weights /= np.sum(weights)
        uv_weights[u_i] = weights  # each line sums to 1.

    return uv_weights


@njit(cache=True)
def _build_nx_filter(directions, sigma_spatial, sigma_align,
                     disable_spatial, disable_align):
    directions = np.ascontiguousarray(directions.astype(np.float32))

    half_width = int(round(3 * sigma_spatial))
    nx_weights = np.zeros((2*half_width+1, 2*half_width+1,
                           2*half_width+1, len(directions)),
                          dtype=np.float32)

    for i in range(-half_width, half_width+1):
        for j in range(-half_width, half_width+1):
            for k in range(-half_width, half_width+1):
                dxy = np.array([[i, j, k]], dtype=np.float32)
                len_xy = np.sqrt(dxy[0, 0]**2 + dxy[0, 1]**2 + dxy[0, 2]**2)

                if disable_spatial:
                    w_spatial = 1.0
                else:
                    # the length controls spatial weight
                    w_spatial = _evaluate_gaussian(sigma_spatial, len_xy)

                # the direction controls the align weight
                if i == j == k == 0 or disable_align:
                    # hack for main direction to have maximal weight
                    # w_align = np.ones((1, len(directions)), dtype=np.float32)
                    w_align = np.zeros((1, len(directions)), dtype=np.float32)
                else:
                    dxy /= len_xy
                    w_align = np.arccos(np.clip(np.dot(dxy, directions.T),
                                                -1.0, 1.0))  # 1, N
                w_align = _evaluate_gaussian(sigma_align, w_align)

                nx_weights[half_width + i, half_width + j, half_width + k] =\
                    w_align * w_spatial

    # sur chaque u, le filtre doit sommer à 1
    for ui in range(len(directions)):
        w_sum = np.sum(nx_weights[..., ui])
        nx_weights /= w_sum

    return nx_weights


@njit(cache=True)
def _evaluate_gaussian(sigma, x):
    # gaussian is not normalized
    return np.exp(-x**2/(2*sigma**2))
