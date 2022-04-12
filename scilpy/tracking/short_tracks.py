# -*- coding: utf-8 -*-
import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
from scilpy.gpuparallel.opencl_utils import CLKernel, CLManager


def track_short_tracks(in_odf, in_seed, in_mask, step_size=0.5,
                       min_length=10., max_length=20.,
                       theta=20.0, batch_size=10000, sh_order=8,
                       sh_basis='descoteaux07'):
    """
    Track short tracks.

    Parameters
    ----------
    in_odf : ndarray
        Spherical harmonics volume. Ex: ODF or fODF.
    in_mask : ndarray
        Tracking mask. Tracking stops outside the mask. Seeding is uniform
        inside the mask.
    step_size : float, optional
        Step size in voxel space. [0.5]
    min_length : float, optional
        Minimum length of a streamline in voxel space. [10]
    max_length : float, optional
        Maximum length of a streamline in voxel space. [20]
    theta : float, optional
        Maximum angle (degrees) between 2 steps. [20.0]
    sh_order : int, optional
        Spherical harmonics order. [8]
    sh_basis : str, optional
        Spherical harmonics basis. ['descoteaux07']

    Returns
    -------
    streamlines: list
        List of short-tracks.
    """
    print('Start short-tracks tracking')

    # Load the sphere
    sphere = get_sphere('symmetric724')
    min_strl_points = int(min_length / step_size) + 1
    max_strl_points = int(max_length / step_size) + 1
    max_cos_theta = np.cos(np.deg2rad(theta))

    cl_kernel = CLKernel('track', 'tracking', 'short_tracks.cl')

    # Set tracking parameters
    cl_kernel.set_define('IM_X_DIM', in_odf.shape[0])
    cl_kernel.set_define('IM_Y_DIM', in_odf.shape[1])
    cl_kernel.set_define('IM_Z_DIM', in_odf.shape[2])
    cl_kernel.set_define('IM_N_COEFFS', in_odf.shape[3])
    cl_kernel.set_define('N_DIRS', len(sphere.vertices))
    cl_kernel.set_define('STEP_SIZE', '{}f'.format(step_size))

    cl_kernel.set_define('MAX_LENGTH', max_strl_points)
    cl_kernel.set_define('MAX_COS_THETA', '{}f'.format(max_cos_theta))

    # Create CL program
    n_input_params = 6
    n_output_params = 2
    cl_manager = CLManager(cl_kernel, n_input_params, n_output_params)

    streamlines = []
    seed_batches = np.array_split(in_seed, len(in_seed)//batch_size)

    # Input buffers
    # Constant input buffers
    cl_manager.add_input_buffer(0, in_odf)
    cl_manager.add_input_buffer(1, sphere.vertices)

    B_mat = sh_to_sf_matrix(sphere, sh_order, sh_basis, return_inv=False)
    cl_manager.add_input_buffer(2, B_mat)
    cl_manager.add_input_buffer(3, in_mask.astype(np.float32))

    # Output buffers
    cl_manager.add_output_buffer(0, (batch_size, max_strl_points, 3))
    cl_manager.add_output_buffer(1, (batch_size, 1))

    # Run the kernel
    # Generate streamlines
    nb_streamlines = 0
    for seed_batch in seed_batches:
        # generate random values for sf sampling
        rand_vals =\
            np.random.uniform(0.0, 1.0, (len(seed_batch), max_strl_points))

        # Update buffers
        cl_manager.add_input_buffer(4, seed_batch)
        cl_manager.add_input_buffer(5, rand_vals)
        cl_manager.add_output_buffer(0, (len(seed_batch), max_strl_points, 3))
        cl_manager.add_output_buffer(1, (len(seed_batch), 1))

        tracks, n_points = cl_manager.run((len(seed_batch), 1, 1))
        n_points = n_points.squeeze().astype(np.int16)
        for (strl, n_pts) in zip(tracks, n_points):
            if n_pts >= min_strl_points:
                streamlines.append(strl[:n_pts])
        nb_streamlines += len(seed_batch)
        print('{0}/{1} streamlines generated'.format(nb_streamlines,
                                                     len(in_seed)))

    print('End short-tracks tracking')
    return streamlines
