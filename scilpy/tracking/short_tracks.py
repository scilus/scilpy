# -*- coding: utf-8 -*-
from time import perf_counter
import logging
import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
from scilpy.gpuparallel.opencl_utils import CLKernel, CLManager


def track_short_tracks(in_odf, in_seed, in_mask,
                       step_size=0.5, min_length=10.,
                       max_length=20., theta=20.0,
                       sharpness=1.0, batch_size=100000,
                       sh_order=8, sh_basis='descoteaux07'):
    """
    Perform probabilistic tracking on a ODF field inside a binary mask. The
    tracking is executed on the GPU using the OpenCL API. Tracking is performed
    in voxel space.

    Streamlines are interrupted as soon as they reach maximum length and
    returned even if they end inside the tracking mask. The ODF image is
    interpolated using nearest neighbor interpolation. No backward tracking is
    performed.

    Parameters
    ----------
    in_odf : ndarray
        Spherical harmonics volume. Ex: ODF or fODF.
    in_seed : ndarray (n_seeds, 3)
        Seed positions.
    in_mask : ndarray
        Tracking mask. Tracking stops outside the mask.
    step_size : float, optional
        Step size in voxel space.
    min_length : float, optional
        Minimum length of a streamline in voxel space.
    max_length : float, optional
        Maximum length of a streamline in voxel space.
    theta : float or list of float, optional
        Maximum angle (degrees) between 2 steps. If a list, a theta
        is randomly drawn from the list for each streamline.
    sharpness : float, optional
        Exponent on ODF amplitude to control sharpness.
    batch_size : int, optional
        Approximate size of GPU batches.
    sh_order : int, optional
        Spherical harmonics order.
    sh_basis : str, optional
        Spherical harmonics basis.

    Returns
    -------
    streamlines: list
        List of streamlines.
    """
    t0 = perf_counter()

    # Load the sphere
    sphere = get_sphere('symmetric724')
    min_strl_points = int(min_length / step_size) + 1
    max_strl_points = int(max_length / step_size) + 1

    # Convert theta to list
    if isinstance(theta, float):
        theta = np.array([theta])
    max_cos_theta = np.cos(np.deg2rad(theta))

    cl_kernel = CLKernel('track', 'tracking', 'short_tracks.cl')

    # Set tracking parameters
    cl_kernel.set_define('IM_X_DIM', in_odf.shape[0])
    cl_kernel.set_define('IM_Y_DIM', in_odf.shape[1])
    cl_kernel.set_define('IM_Z_DIM', in_odf.shape[2])
    cl_kernel.set_define('IM_N_COEFFS', in_odf.shape[3])
    cl_kernel.set_define('N_DIRS', len(sphere.vertices))

    cl_kernel.set_define('N_THETAS', len(theta))
    cl_kernel.set_define('STEP_SIZE', '{}f'.format(step_size))
    cl_kernel.set_define('SHARPEN_ODF_FACTOR', '{}f'.format(sharpness))
    cl_kernel.set_define('MAX_LENGTH', max_strl_points)

    # Create CL program
    n_input_params = 7
    n_output_params = 2
    cl_manager = CLManager(cl_kernel, n_input_params, n_output_params)

    seed_batches = np.array_split(in_seed, np.ceil(len(in_seed)/batch_size))

    # Input buffers
    # Constant input buffers
    cl_manager.add_input_buffer(0, in_odf)
    cl_manager.add_input_buffer(1, sphere.vertices)

    B_mat = sh_to_sf_matrix(sphere, sh_order, sh_basis, return_inv=False)
    cl_manager.add_input_buffer(2, B_mat)
    cl_manager.add_input_buffer(3, in_mask.astype(np.float32))

    cl_manager.add_input_buffer(6, max_cos_theta)

    # Output buffers
    cl_manager.add_output_buffer(0, (batch_size, max_strl_points, 3))
    cl_manager.add_output_buffer(1, (batch_size, 1))
    logging.debug('Initialized OpenCL program in {:.2f}s.'
                  .format(perf_counter() - t0))

    # Generate streamlines in batches
    t0 = perf_counter()
    nb_streamlines = 0
    streamlines = []
    seeds = []
    for seed_batch in seed_batches:
        # generate random values for sf sampling
        rand_vals =\
            np.random.uniform(0.0, 1.0, (len(seed_batch), max_strl_points))

        # Update buffers
        cl_manager.add_input_buffer(4, seed_batch)
        cl_manager.add_input_buffer(5, rand_vals)
        cl_manager.add_output_buffer(0, (len(seed_batch), max_strl_points, 3))
        cl_manager.add_output_buffer(1, (len(seed_batch), 1))

        # Run the kernel
        tracks, n_points = cl_manager.run((len(seed_batch), 1, 1))
        n_points = n_points.squeeze().astype(np.int16)
        for (strl, seed, n_pts) in zip(tracks, seed_batch, n_points):
            if n_pts >= min_strl_points:
                # shift to origin center
                streamlines.append(strl[:n_pts] - 0.5)
                seeds.append(seed - 0.5)
        nb_streamlines += len(seed_batch)
        logging.info('{0:>8}/{1} streamlines generated'
                     .format(nb_streamlines, len(in_seed)))

    logging.info('Tracked {0} streamlines in {1:.2f}s.'
                 .format(len(streamlines), perf_counter() - t0))
    return streamlines, seeds
