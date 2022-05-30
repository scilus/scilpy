# -*- coding: utf-8 -*-
import itertools
import logging
import multiprocessing
import os
import sys
import traceback

import nibabel as nib
import numpy as np

from dipy.tracking.streamlinespeed import compress_streamlines

from scilpy.image.datasets import DataVolume
from scilpy.tracking.propagator import AbstractPropagator, PropagationStatus
from scilpy.tracking.seed import SeedGenerator

# For the multi-processing:
# Iterable. Will contain all parameters necessary for a sub-process
# initialization.
multiprocess_init_args = []


class Tracker(object):
    printing_frequency = 1000

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
        # toDo
        # must be better designed for dipy
        # the tracking should not know which data to deal with
        data_file_name = os.path.join(tmpdir, 'data.npy')
        np.save(data_file_name, self.propagator.dataset.data)

        # Clear data from memory
        self.propagator.reset_data(new_data=None)

        pool = multiprocessing.Pool(
            self.nbr_processes,
            initializer=self._send_multiprocess_args_to_global,
            initargs=(data_file_name, self.mmap_mode))

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
            init_args[0], mmap_mode=init_args[1]))

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
                self.propagator.propagate(line[-1], tracking_info)

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
