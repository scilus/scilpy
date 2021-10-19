# -*- coding: utf-8 -*-
import itertools
import logging
import multiprocessing
import os
import sys
import traceback
import warnings

import nibabel as nib
import numpy as np

from dipy.tracking.streamlinespeed import compress_streamlines

from scilpy.image.datasets import DataVolume
from scilpy.tracking.propagator import AbstractPropagator
from scilpy.tracking.seed import SeedGenerator
from scilpy.tracking.utils import TrackingParams

# For the multi-processing:
data_file_info = None


class Tracker(object):
    def __init__(self, propagator: AbstractPropagator, mask: DataVolume,
                 seed_generator: SeedGenerator, params: TrackingParams,
                 compression_th=0.1, nbr_processes=1, save_seeds=False):
        """
        Parameters
        ----------
        propagator : AbstractPropagator
            Tracking object.
        mask : DataVolume
            Tracking volume(s).
        seed_generator : SeedGenerator
            Seeding volume.
        params: TrackingParams
            Tracking parameters, see scilpy.tracking.utils.py.
        compression_th : float,
            Maximal distance threshold for compression. If None or 0, no
            compression is applied.
        nbr_processes: int
            Number of sub processes to use.
        save_seeds: bool
            Whether to save the seeds associated to their respective
            streamlines.
        """
        self.propagator = propagator
        self.mask = mask
        self.seed_generator = seed_generator
        self.params = params
        self.compression_th = compression_th
        self.save_seeds = save_seeds

        self.nbr_processes = self._set_nbr_processes(nbr_processes)

    def track(self):
        """
        Generate a set of streamline from seed, mask and odf files.

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
                                                      self.params.mmap_mode))

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
                warnings.warn("Cannot determine number of cpus. \
                        returns nbr_processes set to 1.")
                nbr_processes = 1

        if nbr_processes > self.params.nbr_seeds:
            nbr_processes = self.params.nbr_seeds
            logging.debug('Setting number of processes to ' +
                          str(nbr_processes) +
                          ' since there were less seeds than processes.')
        return nbr_processes

    @staticmethod
    def _init_sub_process(date_file_name, mmap_mod):
        global data_file_info
        data_file_info = (date_file_name, mmap_mod)
        return

    def _get_streamlines_sub(self, args):
        """
        multiprocessing.pool.map input function.

        Parameters
        ----------
        args : List, parameters for the get_streamlines_at_seeds function.

        Return
        -------
        lines: list, list of list of 3D positions (streamlines).
        """
        global data_file_info

        # args[0] is the Tracker.
        args[0].tracking_field.dataset.data = np.load(
            data_file_info[0], mmap_mode=data_file_info[1])

        try:
            streamlines, seeds = self._get_streamlines(*args)
            return streamlines, seeds
        except Exception as e:
            logging.error("Operation _get_streamlines_sub() failed.")
            traceback.print_exception(*sys.exc_info(), file=sys.stderr)
            raise e

    def _get_streamlines(self, chunk_id):
        streamlines = []
        seeds = []
        # Initialize the random number generator to cover multiprocessing,
        # skip, which voxel to seed and the subvoxel random position
        chunk_size = int(self.params.nbr_seeds / self.nbr_processes)
        skip = self.params.skip

        first_seed_of_chunk = chunk_id * chunk_size + skip
        random_generator, indices = self.seed_generator.init_pos(
            self.params.random, first_seed_of_chunk)

        if chunk_id == self.nbr_processes - 1:
            chunk_size += self.params.nbr_seeds % self.nbr_processes
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
        np.random.seed(np.uint32(hash((pos, self.params.random))))
        line = []
        if self.propagator.initialize(pos):
            forward = self._propagate_line(True)
            if forward is not None and len(forward) > 0:
                line.extend(forward)

            if not self.params.is_single_direction and forward is not None:
                backward = self._propagate_line(False)
                if backward is not None and len(backward) > 0:
                    line.reverse()
                    line.pop()
                    line.extend(backward)
            else:
                backward = []

            if ((len(line) > 1 and
                 forward is not None and
                 backward is not None and
                 self.params.min_nbr_pts <= len(line) <=
                 self.params.max_nbr_pts)):
                return line
            elif (self.params.is_keep_single_pts and
                  self.params.min_nbr_pts == 1):
                return [pos]
            return None
        if ((self.params.is_keep_single_pts and
             self.params.min_nbr_pts == 1)):
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
        line_dirs = [self.propagator.forward_dir] if is_forward else [
            self.propagator.backward_dir]

        no_valid_direction_count = 0

        propagation_can_continue = True
        while len(line) < self.params.max_nbr_pts and propagation_can_continue:
            new_pos, new_dir, is_valid_direction = self.propagator.propagate(
                line[-1], line_dirs[-1])
            line.append(new_pos)
            line_dirs.append(new_dir)

            if is_valid_direction:
                no_valid_direction_count = 0
            else:
                no_valid_direction_count += 1

            if no_valid_direction_count > self.params.max_no_dir:
                return line

            propagation_can_continue = (
                    self.mask.get_position_value(*line[-1]) > 0 and
                    self.mask.is_position_in_bound(*line[-1]))

        # Make a last step in the last direction
        line.append(line[-1] +
                    self.propagator.step_size * np.array(line_dirs[-1]))

        # Last cleaning of the streamline
        while (line is not None and len(line) > 0 and
               not self.propagator.is_position_in_bound(line[-1])):
            line.pop()

        return line
