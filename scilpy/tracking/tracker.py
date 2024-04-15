# -*- coding: utf-8 -*-
from contextlib import nullcontext
import itertools
import logging
import multiprocessing
import os
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import traceback
from typing import Union
from tqdm import tqdm

import numpy as np
from dipy.data import get_sphere
from dipy.core.sphere import HemiSphere
from dipy.io.stateful_tractogram import Space
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.tracking.streamlinespeed import compress_streamlines

from scilpy.image.volume_space_management import DataVolume
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
                 nbr_processes=1, save_seeds=False,
                 mmap_mode: Union[str, None] = None, rng_seed=1234,
                 track_forward_only=False, skip=0, verbose=False,
                 append_last_point=True):
        """
        Parameters
        ----------
        propagator : AbstractPropagator
            Tracking object.
            This tracker will use space and origin defined in the
            propagator.
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
            Maximal distance threshold for compression. If None, no
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
        verbose: bool
            Display tracking progression.
        append_last_point: bool
            Whether to add the last point (once out of the tracking mask) to
            the streamline or not. Note that points obtained after an invalid
            direction (based on the propagator's definition of invalid; ex
            when angle is too sharp of sh_threshold not reached) are never
            added.
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
        self.append_last_point = append_last_point
        self.skip = skip

        self.origin = self.propagator.origin
        self.space = self.propagator.space
        if self.space == Space.RASMM:
            raise NotImplementedError(
                "This version of the Tracker is not ready to work in RASMM "
                "space.")
        if (seed_generator.origin != propagator.origin or
                seed_generator.space != propagator.space):
            raise ValueError("Seed generator and propagator must work with "
                             "the same space and origin!")

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
        self.verbose = verbose

    def track(self):
        """
        Generate a set of streamline from seed, mask and odf files.

        Return
        ------
        streamlines: list of numpy.array
            List of streamlines, represented as an array of positions.
        seeds: list of numpy.array
            List of seeding positions, one 3-dimensional position per
            streamline.
        """
        if self.nbr_processes < 2:
            chunk_id = 0
            lines, seeds = self._get_streamlines(chunk_id)
        else:
            # Each process will use get_streamlines_at_seeds
            chunk_ids = np.arange(self.nbr_processes)
            with TemporaryDirectory() as tmpdir:
                # Lock for logging
                lock = multiprocessing.Manager().Lock()
                zipped_chunks = zip(chunk_ids, [lock] * self.nbr_processes)

                pool = self._prepare_multiprocessing_pool(tmpdir)

                lines_per_process, seeds_per_process = zip(*pool.map(
                    self._get_streamlines_sub, zipped_chunks))
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
            logging.info("Setting number of processes to {} since there were "
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
        np.save(data_file_name, self.propagator.datavolume.data)

        # Clear data from memory
        self.propagator.reset_data(new_data=None)

        pool = multiprocessing.Pool(
            self.nbr_processes,
            initializer=self._send_multiprocess_args_to_global,
            initargs=({
                'data_file_name': data_file_name,
                'mmap_mode': self.mmap_mode
            },))

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

    def _get_streamlines_sub(self, params):
        """
        multiprocessing.pool.map input function. Calls the main tracking
        method (_get_streamlines) with correct initialization arguments
        (taken from the global variable multiprocess_init_args).

        Parameters
        ----------
        params: Tuple[chunk_id, Lock]
            chunk_id: int, this processes's id.
            Lock: the multiprocessing lock.

        Return
        -------
        lines: list
            List of list of 3D positions (streamlines).
        """
        chunk_id, lock = params
        global multiprocess_init_args

        self._reload_data_for_new_process(multiprocess_init_args)
        try:
            streamlines, seeds = self._get_streamlines(chunk_id, lock)
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

    def _get_streamlines(self, chunk_id, lock=None):
        """
        Tracks the n streamlines associates with current process (identified by
        chunk_id). The number n is the total number of seeds / the number of
        processes. If asked by user, may compress the streamlines and save the
        seeds.

        Parameters
        ----------
        chunk_id: int
            This process ID.
        lock: Lock
            The multiprocessing lock for verbose printing (optional with
            single processing).

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
        tqdm_text = "#" + "{}".format(chunk_id).zfill(3)

        if self.verbose:
            if lock is None:
                lock = nullcontext()
            with lock:
                # Note. Option miniters does not work with manual pbar update.
                # Will verify manually, lower.
                # Fixed choice of value rather than a percentage of the chunk
                # size because our tracker is quite slow.
                miniters = 100
                p = tqdm(total=chunk_size, desc=tqdm_text, position=chunk_id+1,
                         leave=False)

        for s in range(chunk_size):
            seed = self.seed_generator.get_next_pos(
                random_generator, indices, first_seed_of_chunk + s)

            # Setting the random value.
            # Previous usage (and usage in Dipy) is to set the random seed
            # based on the (real) seed position. However, in the case where we
            # like to have exactly the same seed more than once, this will lead
            # to exactly the same line, even in probabilistic tracking.
            # Changing to seed position + seed number.
            # Then in the case of multiprocessing, adding also a fraction based
            # on current process ID.
            eps = s + chunk_id / (self.nbr_processes + 1)
            line_generator = np.random.default_rng(
                np.uint32(hash((seed + (eps, eps, eps), self.rng_seed))))

            # Forward and backward tracking
            line = self._get_line_both_directions(seed, line_generator)

            if line is not None:
                streamline = np.array(line, dtype='float32')

                if self.compression_th is not None:
                    # Compressing. Threshold is in mm. Verifying space.
                    if self.space == Space.VOX:
                        # Equivalent of sft.to_voxmm:
                        streamline *= self.seed_generator.voxres
                        compress_streamlines(streamline, self.compression_th)
                        # Equivalent of sft.to_vox:
                        streamline /= self.seed_generator.voxres
                    else:
                        compress_streamlines(streamline, self.compression_th)

                streamlines.append(streamline)

                if self.save_seeds:
                    seeds.append(np.asarray(seed, dtype='float32'))

            if self.verbose and (s + 1) % miniters == 0:
                with lock:
                    p.update(miniters)

        if self.verbose:
            with lock:
                p.close()
        return streamlines, seeds

    def _get_line_both_directions(self, seeding_pos, line_generator):
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
        # Forward
        line = [np.asarray(seeding_pos)]
        tracking_info = self.propagator.prepare_forward(seeding_pos,
                                                        line_generator)
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

            # Verifying if direction is valid
            # If invalid: break. Else, verify tracking mask.
            if is_direction_valid:
                invalid_direction_count = 0
            else:
                invalid_direction_count += 1
                if invalid_direction_count > self.max_invalid_dirs:
                    break

            propagation_can_continue = self._verify_stopping_criteria(new_pos)
            if propagation_can_continue or self.append_last_point:
                line.append(new_pos)

            tracking_info = new_tracking_info

        return line

    def _verify_stopping_criteria(self, last_pos):

        # Checking if out of bound
        if not self.mask.is_coordinate_in_bound(
                *last_pos, space=self.space, origin=self.origin):
            return False

        # Checking if out of mask
        if self.mask.get_value_at_coordinate(
                *last_pos, space=self.space, origin=self.origin) <= 0:
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
        Seed positions in voxel space with origin `center`.
    step_size : float
        Step size in voxel space.
    max_nbr_pts : int
        Maximum length of a streamline in voxel space.
    theta : float or list of float, optional
        Maximum angle (degrees) between 2 steps. If a list, a theta
        is randomly drawn from the list for each streamline.
    sh_basis : str, optional
        Spherical harmonics basis.
    is_legacy : bool, optional
        Whether or not the SH basis is in its legacy form.
    batch_size : int, optional
        Approximate size of GPU batches.
    forward_only: bool, optional
        If True, only forward tracking is performed.
    rng_seed : int, optional
        Seed for random number generator.
    sphere : int, optional
        Sphere to use for the tracking.
    """
    def __init__(self, sh, mask, seeds, step_size, max_nbr_pts,
                 theta=20.0, sf_threshold=0.1, sh_interp='trilinear',
                 sh_basis='descoteaux07', is_legacy=True, batch_size=100000,
                 forward_only=False, rng_seed=None, sphere=None):
        if not have_opencl:
            raise ImportError('pyopencl is not installed. In order to use'
                              'GPU tracker, you need to install it first.')
        self.sh = sh
        if sh_interp not in ['nearest', 'trilinear']:
            raise ValueError('Invalid SH interpolation mode: {}'
                             .format(sh_interp))
        self.sh_interp_nn = sh_interp == 'nearest'
        self.mask = mask

        self.n_seeds = len(seeds)

        self.seed_batches =\
            np.array_split(seeds + 0.5, np.ceil(len(seeds)/batch_size))

        if sphere is None:
            self.sphere = get_sphere("repulsion724")
        else:
            self.sphere = sphere

        # tracking step_size and number of points
        self.step_size = step_size
        self.sf_threshold = sf_threshold
        self.max_strl_points = max_nbr_pts

        # convert theta to array
        self.theta = np.atleast_1d(theta)

        self.sh_basis = sh_basis
        self.is_legacy = is_legacy
        self.forward_only = forward_only

        # Instantiate random number generator
        self.rng = np.random.default_rng(rng_seed)

    def _get_max_amplitudes(self, B_mat):
        fodf_max = np.zeros(self.mask.shape,
                            dtype=np.float32)
        fodf_max[self.mask > 0] = np.max(self.sh[self.mask > 0].dot(B_mat),
                                         axis=-1)

        return fodf_max

    def __iter__(self):
        return self._track()

    def _track(self):
        """
        GPU streamlines generator yielding streamlines with corresponding
        seed positions one by one.
        """
        # Convert theta to cos(theta)
        max_cos_theta = np.cos(np.deg2rad(self.theta))

        cl_kernel = CLKernel('tracker', 'tracking', 'local_tracking.cl')

        # Set tracking parameters
        cl_kernel.set_define('IM_X_DIM', self.sh.shape[0])
        cl_kernel.set_define('IM_Y_DIM', self.sh.shape[1])
        cl_kernel.set_define('IM_Z_DIM', self.sh.shape[2])
        cl_kernel.set_define('IM_N_COEFFS', self.sh.shape[3])
        cl_kernel.set_define('N_DIRS', len(self.sphere.vertices))

        cl_kernel.set_define('N_THETAS', len(self.theta))
        cl_kernel.set_define('STEP_SIZE', '{:.8f}f'.format(self.step_size))
        cl_kernel.set_define('MAX_LENGTH', self.max_strl_points)
        cl_kernel.set_define('FORWARD_ONLY',
                             'true' if self.forward_only else 'false')
        cl_kernel.set_define('SF_THRESHOLD',
                             '{:.8f}f'.format(self.sf_threshold))
        cl_kernel.set_define('SH_INTERP_NN',
                             'true' if self.sh_interp_nn else 'false')

        # Create CL program
        cl_manager = CLManager(cl_kernel)

        # Input buffers
        # Constant input buffers
        cl_manager.add_input_buffer('sh', self.sh)
        cl_manager.add_input_buffer('vertices', self.sphere.vertices)

        sh_order = find_order_from_nb_coeff(self.sh)
        B_mat = sh_to_sf_matrix(self.sphere, sh_order, self.sh_basis,
                                return_inv=False, legacy=self.is_legacy)
        cl_manager.add_input_buffer('b_matrix', B_mat)

        fodf_max = self._get_max_amplitudes(B_mat)
        cl_manager.add_input_buffer('max_amplitudes', fodf_max)
        cl_manager.add_input_buffer('mask', self.mask.astype(np.float32))

        cl_manager.add_input_buffer('max_cos_theta', max_cos_theta)

        cl_manager.add_input_buffer('seeds')
        cl_manager.add_input_buffer('randvals')

        cl_manager.add_output_buffer('out_strl')
        cl_manager.add_output_buffer('out_lengths')

        # Generate streamlines in batches
        for seed_batch in self.seed_batches:
            # Generate random values for sf sampling
            # TODO: Implement random number generator directly
            #       on the GPU to generate values on-the-fly.
            rand_vals = self.rng.uniform(0.0, 1.0,
                                         (len(seed_batch),
                                          self.max_strl_points))

            # Update buffers
            cl_manager.update_input_buffer('seeds', seed_batch)
            cl_manager.update_input_buffer('randvals', rand_vals)

            # output streamlines buffer
            cl_manager.update_output_buffer('out_strl',
                                            (len(seed_batch),
                                             self.max_strl_points, 3))
            # output streamlines length buffer
            cl_manager.update_output_buffer('out_lengths',
                                            (len(seed_batch), 1))

            # Run the kernel
            tracks, n_points = cl_manager.run((len(seed_batch), 1, 1))
            n_points = n_points.flatten().astype(np.int16)
            for (strl, seed, n_pts) in zip(tracks, seed_batch, n_points):
                strl = strl[:n_pts]

                # output is yielded so that we can use LazyTractogram.
                # seed and strl with origin center (same as DIPY)
                yield strl - 0.5, seed - 0.5
