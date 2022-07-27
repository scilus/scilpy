# -*- coding: utf-8 -*-
import itertools
import logging
import multiprocessing
import os
import sys
import traceback
from time import perf_counter

import nibabel as nib
import numpy as np

from dipy.tracking.streamlinespeed import compress_streamlines
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix

from scilpy.image.datasets import DataVolume
from scilpy.tracking.propagator import AbstractPropagator, PropagationStatus
from scilpy.reconst.utils import find_order_from_nb_coeff
from scilpy.tracking.seed import SeedGenerator
from scilpy.gpuparallel.opencl_utils import CLKernel, CLManager, have_opencl

# For the multi-processing:
# Dictionary. Will contain all parameters necessary for a sub-process
# initialization.
multiprocess_init_args = {}


class Tracker(object):
    def __init__(self, propagator: AbstractPropagator, mask: DataVolume,
                 seed_generator: SeedGenerator, nbr_seeds, min_nbr_pts,
                 max_nbr_pts, max_invalid_dirs, compression_th=0.1,
                 nbr_processes=1, save_seeds=False, mmap_mode=None,
                 rng_seed=1234, track_forward_only=False, skip=0):
        """
        Parameters
        ----------
        propagator : AbstractPropagator
            Tracking object.
        mask : DataVolume
            Tracking volume(s).
        seed_generator : SeedGenerator
            Seeding volume.
        nbr_seeds: int
            Number of seeds to create via the seed generator.
        min_nbr_pts: int
            Minimum number of points for streamlines.
        max_nbr_pts: int
            Maximum number of points for streamlines.
        max_invalid_dirs: int
            Number of consecutives invalid directions allowed during tracking.
        compression_th : float,
            Maximal distance threshold for compression. If None or 0, no
            compression is applied.
        nbr_processes: int
            Number of sub processes to use.
        save_seeds: bool
            Whether to save the seeds associated to their respective
            streamlines.
        mmap_mode: str
            Memory-mapping mode. One of {None, 'r+', 'c'}. This value is passed
            to np.load() when loading the raw tracking data from a subprocess.
        rng_seed: int
            The random "seed" for the random generator.
        track_forward_only: bool
            If true, only the forward direction is computed.
        skip: int
            Skip the first N seeds created (and thus N rng numbers). Useful if
            you want to create new streamlines to add to a previously created
            tractogram with a fixed rng_seed. Ex: If tractogram_1 was created
            with nbr_seeds=1,000,000, you can create tractogram_2 with
            skip 1,000,000.
        """
        self.propagator = propagator
        self.mask = mask
        self.seed_generator = seed_generator
        self.nbr_seeds = nbr_seeds
        self.min_nbr_pts = min_nbr_pts
        self.max_nbr_pts = max_nbr_pts
        self.max_invalid_dirs = max_invalid_dirs
        self.compression_th = compression_th
        self.save_seeds = save_seeds
        self.mmap_mode = mmap_mode
        self.rng_seed = rng_seed
        self.track_forward_only = track_forward_only
        self.skip = skip

        # Everything scilpy.tracking is in 'corner', 'voxmm'
        self.origin = 'corner'
        self.space = 'voxmm'

        if self.min_nbr_pts <= 0:
            logging.warning("Minimum number of points cannot be 0. Changed to "
                            "1.")
            self.min_nbr_pts = 1

        if self.mmap_mode not in [None, 'r+', 'c']:
            logging.warning("Memory-mapping mode cannot be {}. Changed to "
                            "None.".format(self.mmap_mode))
            self.mmap_mode = None

        self.nbr_processes = self._set_nbr_processes(nbr_processes)

        self.printing_frequency = 1000

    def track(self):
        """
        Generate a set of streamline from seed, mask and odf files. Results
        are in voxmm space (i.e. in mm coordinates, starting at 0,0,0).

        Return
        ------
        streamlines: list of numpy.array
            List of streamlines, represented as an array of positions.
        seeds: list of numpy.array
            List of seeding positions, one 3-dimensional position per
            streamline.
        """
        if self.nbr_processes < 2:
            chunk_id = 1
            lines, seeds = self._get_streamlines(chunk_id)
        else:
            # Each process will use get_streamlines_at_seeds
            chunk_ids = np.arange(self.nbr_processes)
            with nib.tmpdirs.InTemporaryDirectory() as tmpdir:

                pool = self._prepare_multiprocessing_pool(tmpdir)

                lines_per_process, seeds_per_process = zip(*pool.map(
                    self._get_streamlines_sub, chunk_ids))
                pool.close()
                # Make sure all worker processes have exited before leaving
                # context manager.
                pool.join()
                lines = [line for line in itertools.chain(*lines_per_process)]
                seeds = [seed for seed in itertools.chain(*seeds_per_process)]

        return lines, seeds

    def _set_nbr_processes(self, nbr_processes):
        """
        If user did not define the number of processes, define it automatically
        (or set to 1 -- no multiprocessing -- if we can't).
        """
        if nbr_processes <= 0:
            try:
                nbr_processes = multiprocessing.cpu_count()
            except NotImplementedError:
                logging.warning("Cannot determine number of cpus: "
                                "nbr_processes set to 1.")
                nbr_processes = 1

        if nbr_processes > self.nbr_seeds:
            nbr_processes = self.nbr_seeds
            logging.debug("Setting number of processes to {} since there were "
                          "less seeds than processes.".format(nbr_processes))
        return nbr_processes

    def _prepare_multiprocessing_pool(self, tmpdir):
        """
        Prepare multiprocessing pool.

        Data must be carefully managed to avoid corruption with
        multiprocessing.

        Params
        ------
        tmpdir: str
            Path where to save temporarily the data. This will allow clearing
            the data from memory. We will fetch it back later.

        Returns
        -------
        pool: The multiprocessing pool.
        """
        # Using pool with a class method will serialize all parameters
        # in the class, which can be heavy, but it is what we would be
        # doing manually with a static class.
        # Be careful however, parameter changes inside the method will
        # not be kept.

        # Saving data. We will reload it in each process.
        data_file_name = os.path.join(tmpdir, 'data.npy')
        np.save(data_file_name, self.propagator.dataset.data)

        # Clear data from memory
        self.propagator.reset_data(new_data=None)

        pool = multiprocessing.Pool(
            self.nbr_processes,
            initializer=self._send_multiprocess_args_to_global,
            initargs={
                'data_file_name': data_file_name,
                'mmap_mode': self.mmap_mode
            })

        return pool

    @staticmethod
    def _send_multiprocess_args_to_global(init_args):
        """
        Sends subprocess' initialisation arguments to global for easier access
        by the multiprocessing pool.
        """
        global multiprocess_init_args
        multiprocess_init_args = init_args
        return

    def _get_streamlines_sub(self, chunk_id):
        """
        multiprocessing.pool.map input function. Calls the main tracking
        method (_get_streamlines) with correct initialization arguments
        (taken from the global variable multiprocess_init_args).

        Parameters
        ----------
        chunk_id: int
            This processes's id.

        Return
        -------
        lines: list
            List of list of 3D positions (streamlines).
        """
        global multiprocess_init_args

        self._reload_data_for_new_process(multiprocess_init_args)
        try:
            streamlines, seeds = self._get_streamlines(chunk_id)
            return streamlines, seeds
        except Exception as e:
            logging.error("Operation _get_streamlines_sub() failed.")
            traceback.print_exception(*sys.exc_info(), file=sys.stderr)
            raise e

    def _reload_data_for_new_process(self, init_args):
        """
        Once process is started, load back data.

        Params
        ------
        init_args: Iterable
            Args necessary to reset data. In current implementation: a tuple;
            (file where the data is saved, mmap_mode).
        """
        self.propagator.reset_data(np.load(
            init_args['data_file_name'], mmap_mode=init_args['mmap_mode']))

    def _get_streamlines(self, chunk_id):
        """
        Tracks the n streamlines associates with current process (identified by
        chunk_id). The number n is the total number of seeds / the number of
        processes. If asked by user, may compress the streamlines and save the
        seeds.

        Parameters
        ----------
        chunk_id: int
            This process ID.

        Returns
        -------
        streamlines: list
            The successful streamlines.
        seeds: list
            The list of seeds for each streamline, if self.save_seeds. Else, an
            empty list.
        """
        streamlines = []
        seeds = []

        # Initialize the random number generator to cover multiprocessing,
        # skip, which voxel to seed and the subvoxel random position
        chunk_size = int(self.nbr_seeds / self.nbr_processes)
        first_seed_of_chunk = chunk_id * chunk_size + self.skip
        random_generator, indices = self.seed_generator.init_generator(
            self.rng_seed, first_seed_of_chunk)
        if chunk_id == self.nbr_processes - 1:
            chunk_size += self.nbr_seeds % self.nbr_processes

        # Getting streamlines
        for s in range(chunk_size):
            if s % self.printing_frequency == 0:
                logging.info("Process {} (id {}): {} / {}"
                             .format(chunk_id, os.getpid(), s, chunk_size))

            seed = self.seed_generator.get_next_pos(
                random_generator, indices, first_seed_of_chunk + s)

            # Forward and backward tracking
            line = self._get_line_both_directions(seed)

            if line is not None:
                if self.compression_th and self.compression_th > 0:
                    streamlines.append(
                        compress_streamlines(np.array(line, dtype='float32'),
                                             self.compression_th))
                else:
                    streamlines.append((np.array(line, dtype='float32')))

                if self.save_seeds:
                    seeds.append(np.asarray(seed, dtype='float32'))
        return streamlines, seeds

    def _get_line_both_directions(self, seeding_pos):
        """
        Generate a streamline from an initial position following the tracking
        parameters.

        Parameters
        ----------
        seeding_pos : tuple
            3D position, the seed position.

        Returns
        -------
        line: list of 3D positions
            The generated streamline for seeding_pos.
        """

        # toDo See numpy's doc: np.random.seed:
        #  This is a convenience, legacy function.
        #  The best practice is to not reseed a BitGenerator, rather to
        #  recreate a new one. This method is here for legacy reasons.
        np.random.seed(np.uint32(hash((seeding_pos, self.rng_seed))))

        # Forward
        line = [np.asarray(seeding_pos)]
        tracking_info = self.propagator.prepare_forward(seeding_pos)
        if tracking_info == PropagationStatus.ERROR:
            # No good tracking direction can be found at seeding position.
            return None
        line = self._propagate_line(line, tracking_info)

        # Backward
        if not self.track_forward_only:
            if len(line) > 1:
                line.reverse()

            tracking_info = self.propagator.prepare_backward(line,
                                                             tracking_info)
            line = self._propagate_line(line, tracking_info)

        # Clean streamline
        if self.min_nbr_pts <= len(line) <= self.max_nbr_pts:
            return line
        return None

    def _propagate_line(self, line, tracking_info):
        """
        Generate a streamline in forward or backward direction from an initial
        position following the tracking parameters.

        Propagation will stop if the current position is out of bounds (mask's
        bounds and data's bounds should be the same) or if mask's value at
        current position is 0 (usual use is with a binary mask but this is not
        mandatory).

        Parameters
        ----------
        line: List[np.ndarrays]
            Beginning of the line to propagate: list of 3D coordinates
            formatted as arrays.
        tracking_info: Any
            Information necessary to know how to propagate. Type: as understood
            by the propagator. Example, with the typical fODF propagator: the
            previous direction of the streamline, v_in, used to define a cone
            theta, of type TrackingDirection.

        Returns
        -------
        line: list of 3D positions
            At minimum, stays as initial line. Or extended with new tracked
            points.
        """
        invalid_direction_count = 0
        propagation_can_continue = True
        while len(line) < self.max_nbr_pts and propagation_can_continue:
            new_pos, new_tracking_info, is_direction_valid = \
                self.propagator.propagate(line, tracking_info)

            # Verifying and appending
            if is_direction_valid:
                invalid_direction_count = 0
            else:
                invalid_direction_count += 1
            propagation_can_continue = self._verify_stopping_criteria(
                invalid_direction_count, new_pos)
            if propagation_can_continue:
                line.append(new_pos)

            tracking_info = new_tracking_info

        # Possible last step.
        final_pos = self.propagator.finalize_streamline(line[-1],
                                                        tracking_info)
        if (final_pos is not None and
                not np.array_equal(final_pos, line[-1]) and
                self.mask.is_voxmm_in_bound(*final_pos, origin=self.origin)):
            line.append(final_pos)
        return line

    def _verify_stopping_criteria(self, invalid_direction_count, last_pos):
        # Checking number of consecutive invalid directions
        if invalid_direction_count > self.max_invalid_dirs:
            return False

        # Checking if out of bound
        if not self.mask.is_voxmm_in_bound(*last_pos, origin=self.origin):
            return False

        # Checking if out of mask
        if self.mask.voxmm_to_value(*last_pos, origin=self.origin) <= 0:
            return False

        return True


class GPUTacker():
    """
    Perform probabilistic tracking on a ODF field inside a binary mask. The
    tracking is executed on the GPU using the OpenCL API. Tracking is performed
    in voxel space with origin `corner`.

    Streamlines are interrupted as soon as they reach maximum length and
    returned even if they end inside the tracking mask. The ODF image is
    interpolated using nearest neighbor interpolation. No backward tracking is
    performed.

    Parameters
    ----------
    sh : ndarray
        Spherical harmonics volume. Ex: ODF or fODF.
    mask : ndarray
        Tracking mask. Tracking stops outside the mask.
    seeds : ndarray (n_seeds, 3)
        Seed positions in voxel space with origin `corner`.
    step_size : float
        Step size in voxel space.
    min_nbr_pts : int
        Minimum length of a streamline in voxel space.
    max_nbr_pts : int
        Maximum length of a streamline in voxel space.
    theta : float or list of float, optional
        Maximum angle (degrees) between 2 steps. If a list, a theta
        is randomly drawn from the list for each streamline.
    sh_basis : str, optional
        Spherical harmonics basis.
    batch_size : int, optional
        Approximate size of GPU batches.
    forward_only: bool, optional
        If True, only forward tracking is performed.
    rng_seed : int, optional
        Seed for random number generator.
    """
    def __init__(self, sh, mask, seeds, step_size, min_nbr_pts, max_nbr_pts,
                 theta=20.0, sh_basis='descoteaux07', batch_size=100000,
                 forward_only=False, rng_seed=None):
        if not have_opencl:
            raise ImportError('pyopencl is not installed. In order to use'
                              'GPU tracker, you need to install it first.')
        self.sh = sh
        self.mask = mask

        if (seeds < 0).any():
            raise ValueError('Invalid seed positions.\nGPUTracker works with'
                             ' origin \'corner\'.')
        self.n_seeds = len(seeds)
        self.seed_batches =\
            np.array_split(seeds, np.ceil(len(seeds)/batch_size))

        # tracking step_size and number of points
        self.step_size = step_size
        self.min_strl_points = min_nbr_pts
        self.max_strl_points = max_nbr_pts

        # convert theta to array
        if isinstance(theta, float):
            theta = np.array([theta])
        self.theta = theta

        self.sh_basis = sh_basis
        self.forward_only = forward_only

        # Instantiate random number generator
        self.rng = np.random.default_rng(rng_seed)

    def track(self):
        """
        GPU streamlines generator yielding streamlines with corresponding
        seed positions one by one.
        """
        t0 = perf_counter()

        # Load the sphere
        sphere = get_sphere('symmetric724')

        # Convert theta to cos(theta)
        max_cos_theta = np.cos(np.deg2rad(self.theta))

        cl_kernel = CLKernel('track', 'tracking', 'local_tracking.cl')

        # Set tracking parameters
        # TODO: Add relative sf_threshold parameter.
        cl_kernel.set_define('IM_X_DIM', self.sh.shape[0])
        cl_kernel.set_define('IM_Y_DIM', self.sh.shape[1])
        cl_kernel.set_define('IM_Z_DIM', self.sh.shape[2])
        cl_kernel.set_define('IM_N_COEFFS', self.sh.shape[3])
        cl_kernel.set_define('N_DIRS', len(sphere.vertices))

        cl_kernel.set_define('N_THETAS', len(self.theta))
        cl_kernel.set_define('STEP_SIZE', '{}f'.format(self.step_size))
        cl_kernel.set_define('MAX_LENGTH', self.max_strl_points)
        cl_kernel.set_define('FORWARD_ONLY',
                             'true' if self.forward_only else 'false')

        # Create CL program
        n_input_params = 7
        n_output_params = 2
        cl_manager = CLManager(cl_kernel, n_input_params, n_output_params)

        # Input buffers
        # Constant input buffers
        cl_manager.add_input_buffer(0, self.sh)
        cl_manager.add_input_buffer(1, sphere.vertices)

        sh_order = find_order_from_nb_coeff(self.sh)
        B_mat = sh_to_sf_matrix(sphere, sh_order, self.sh_basis,
                                return_inv=False)
        cl_manager.add_input_buffer(2, B_mat)
        cl_manager.add_input_buffer(3, self.mask.astype(np.float32))

        cl_manager.add_input_buffer(6, max_cos_theta)

        logging.debug('Initialized OpenCL program in {:.2f}s.'
                      .format(perf_counter() - t0))

        # Generate streamlines in batches
        t0 = perf_counter()
        nb_processed_streamlines = 0
        nb_valid_streamlines = 0
        for seed_batch in self.seed_batches:
            # Generate random values for sf sampling
            # TODO: Implement random number generator directly
            #       on the GPU to generate values on-the-fly.
            rand_vals = self.rng.uniform(0.0, 1.0,
                                         (len(seed_batch),
                                          self.max_strl_points))

            # Update buffers
            cl_manager.add_input_buffer(4, seed_batch)
            cl_manager.add_input_buffer(5, rand_vals)

            # output streamlines buffer
            cl_manager.add_output_buffer(
                0, (len(seed_batch), self.max_strl_points, 3))
            # output streamlines length buffer
            cl_manager.add_output_buffer(1, (len(seed_batch), 1))

            # Run the kernel
            tracks, n_points = cl_manager.run((len(seed_batch), 1, 1))
            n_points = n_points.squeeze().astype(np.int16)
            for (strl, seed, n_pts) in zip(tracks, seed_batch, n_points):
                if n_pts >= self.min_strl_points:
                    strl = strl[:n_pts]
                    nb_valid_streamlines += 1

                    # output is yielded so that we can use lazy tractogram.
                    yield strl, seed

            # per-batch logging information
            nb_processed_streamlines += len(seed_batch)
            logging.info('{0:>8}/{1} streamlines generated'
                         .format(nb_processed_streamlines, self.n_seeds))

        logging.info('Tracked {0} streamlines in {1:.2f}s.'
                     .format(nb_valid_streamlines, perf_counter() - t0))
