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
from scilpy.tracking.propagator import AbstractPropagator
from scilpy.tracking.seed import SeedGenerator

# For the multi-processing:
data_file_info = None


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
        max_nbr_pts: int
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
            Memory-mapping mode. One of {None, ‘r+’, ‘r’, ‘w+’, ‘c’}
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

        if self.min_nbr_pts <= 0:
            logging.warning("Minimum number of points cannot be 0. Changed to "
                            "1.")
            self.min_nbr_pts = 1

        self.nbr_processes = self._set_nbr_processes(nbr_processes)

    def track(self):
        """
        Generate a set of streamline from seed, mask and odf files. Results
        are in voxmm space (i.e. in mm coordinates, starting at 0,0,0).

        Return
        ------
        streamlines: list of numpy.array
        seeds: list of numpy.array
        """
        if self.nbr_processes < 2:
            chunk_id = 1
            lines, seeds = self._get_streamlines(chunk_id)
        else:
            # Each process will use get_streamlines_at_seeds
            chunk_ids = np.arange(self.nbr_processes)
            with nib.tmpdirs.InTemporaryDirectory() as tmpdir:
                # toDo
                # must be better designed for dipy
                # the tracking should not know which data to deal with
                data_file_name = os.path.join(tmpdir, 'data.npy')
                np.save(data_file_name,
                        self.propagator.tracking_field.dataset.data)
                self.propagator.tracking_field.dataset.data = None

                # Using pool with a class method will serialize all parameters
                # in the class, which can be heavy, but it is what we would be
                # doing manually with a static class.
                # Be careful however, parameter changes inside the method will
                # not be kept.
                pool = multiprocessing.Pool(self.nbr_processes,
                                            initializer=self._init_sub_process,
                                            initargs=(data_file_name,
                                                      self.mmap_mode))

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
        # Verifying the number of processes
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

    @staticmethod
    def _init_sub_process(date_file_name, mmap_mod):
        global data_file_info
        data_file_info = (date_file_name, mmap_mod)
        return

    def _get_streamlines_sub(self, chunk_id):
        """
        multiprocessing.pool.map input function.

        Parameters
        ----------
        chunk_id: int, This processes's id.

        Return
        -------
        lines: list, list of list of 3D positions (streamlines).
        """
        global data_file_info

        # args[0] is the Tracker.
        self.propagator.tracking_field.dataset.data = np.load(
            data_file_info[0], mmap_mode=data_file_info[1])

        try:
            streamlines, seeds = self._get_streamlines(chunk_id)
            return streamlines, seeds
        except Exception as e:
            logging.error("Operation _get_streamlines_sub() failed.")
            traceback.print_exception(*sys.exc_info(), file=sys.stderr)
            raise e

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
            if s % 1000 == 0:
                logging.info(str(os.getpid()) + " : " + str(s)
                             + " / " + str(chunk_size))

            seed = self.seed_generator.get_next_pos(
                random_generator, indices, first_seed_of_chunk + s)
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

    def _get_line_both_directions(self, pos):
        """
        Generate a streamline from an initial position following the tracking
        parameters.

        Parameters
        ----------
        pos : tuple
            3D position, the seed position.

        Returns
        -------
        line: list of 3D positions
        """

        # toDo See numpy's doc: np.random.seed:
        #  This is a convenience, legacy function.
        #  The best practice is to not reseed a BitGenerator, rather to
        #  recreate a new one. This method is here for legacy reasons.
        np.random.seed(np.uint32(hash((pos, self.rng_seed))))
        line = [pos]

        # Initialize returns true if initial directions at pos are valid.
        if self.propagator.initialize(pos, self.track_forward_only):
            # Forward
            forward = self._propagate_line(True)
            if len(forward) > 0:
                forward.pop(0)
                line.extend(forward)

            # Backward
            if not self.track_forward_only:
                backward = self._propagate_line(False)
                if len(backward) > 0:
                    line.reverse()
                    line.pop()
                    line.extend(backward)

            # Clean streamline
            if self.min_nbr_pts <= len(line) <= self.max_nbr_pts:
                return line
            return None
        elif self.min_nbr_pts == 1:
            return [pos]
        return None

    def _propagate_line(self, is_forward):
        """
        Generate a streamline in forward or backward direction from an initial
        position following the tracking parameters.

        Propagation will stop if the current position is out of bounds (mask's
        bounds and data's bounds should be the same) or if mask's value at
        current position is 0 (usual use is with a binary mask but this is not
        mandatory).

        Returns
        -------
        line: list of 3D positions
        """
        line = [self.propagator.init_pos]
        last_dir = self.propagator.forward_dir if is_forward else \
            self.propagator.backward_dir

        invalid_direction_count = 0

        propagation_can_continue = True
        while len(line) < self.max_nbr_pts and propagation_can_continue:
            new_pos, new_dir, is_valid_direction = self.propagator.propagate(
                line[-1], last_dir)
            line.append(new_pos)

            if is_valid_direction:
                invalid_direction_count = 0
            else:
                invalid_direction_count += 1

            if invalid_direction_count > self.max_invalid_dirs:
                propagation_can_continue = False
                break

            # Bound can be checked with mask or tracking field
            # (through self.propagator.is_voxmm_in_bound)
            propagation_can_continue = (
                    self.mask.voxmm_to_value(*line[-1]) > 0 and
                    self.mask.is_voxmm_in_bound(*line[-1], origin='corner'))
            last_dir = new_dir

        if propagation_can_continue:
            # Make a last step in the last direction
            # Ex: if mask is WM, reaching GM a little more.
            line.append(line[-1] +
                        self.propagator.step_size * np.array(last_dir))

        # Last cleaning of the streamline
        # First position is the seed: necessarily in bound.
        while (len(line) > 1 and
               not self.propagator.is_voxmm_in_bound(line[-1], 'corner')):
            line.pop()

        return line
