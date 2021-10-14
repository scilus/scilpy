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

from scilpy.image.datasets import AccessibleVolume
from scilpy.tracking.tracker import AbstractTracker
from scilpy.tracking.seed import SeedGenerator
from scilpy.tracking.utils import TrackingParams

data_file_info = None


def track(tracker: AbstractTracker, mask: AccessibleVolume,
          seed_generator: SeedGenerator, params: TrackingParams,
          compression_th=0.1, nbr_processes=1, save_seeds=False):
    """
    Generate a set of streamline from seed, mask and odf files.

    Parameters
    ----------
    tracker : AbstractTracker
        Tracking object.
    mask : AccessibleVolume
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
        Whether to save the seeds associated to their respective streamlines.

    Return
    ------
    streamlines: list of numpy.array
    seeds: list of numpy.array
    """
    # Verifying the number of processes
    if nbr_processes <= 0:
        try:
            nbr_processes = multiprocessing.cpu_count()
        except NotImplementedError:
            warnings.warn("Cannot determine number of cpus. \
                returns nbr_processes set to 1.")
            nbr_processes = 1

    if nbr_processes > params.nbr_seeds:
        nbr_processes = params.nbr_seeds
        logging.debug('Setting number of processes to ' +
                      str(nbr_processes) +
                      ' since there were less seeds than processes.')
    if nbr_processes < 2:
        chunk_id = 1
        lines, seeds = get_streamlines_at_seeds(
            tracker, mask, seed_generator, chunk_id, params, compression_th,
            nbr_processes=1, save_seeds=save_seeds)
    else:
        # Each process will use get_streamlines_at_seeds
        chunk_ids = np.arange(nbr_processes)
        with nib.tmpdirs.InTemporaryDirectory() as tmpdir:
            # toDo
            # must be better designed for dipy
            # the tracking should not know which data to deal with
            data_file_name = os.path.join(tmpdir, 'data.npy')
            np.save(data_file_name, tracker.tracking_field.dataset.data)
            tracker.tracking_field.dataset.data = None

            pool = multiprocessing.Pool(nbr_processes,
                                        initializer=_init_sub_process,
                                        initargs=(data_file_name,
                                                  params.mmap_mode))

            lines_per_process, seeds_per_process = zip(*pool.map(
                _get_streamlines_at_seeds_sub,
                zip(itertools.repeat(tracker),
                    itertools.repeat(mask),
                    itertools.repeat(seed_generator),
                    chunk_ids,
                    itertools.repeat(params),
                    itertools.repeat(compression_th),
                    itertools.repeat(nbr_processes),
                    itertools.repeat(save_seeds))))
            pool.close()
            # Make sure all worker processes have exited before leaving
            # context manager in order to prevent temporary file deletion
            # errors in Windows
            pool.join()
            lines = np.array([line for line in
                              itertools.chain(*lines_per_process)])
            seeds = np.array([seed for seed in
                              itertools.chain(*seeds_per_process)])

    return lines, seeds


def _init_sub_process(date_file_name, mmap_mod):
    global data_file_info
    data_file_info = (date_file_name, mmap_mod)
    return


def _get_streamlines_at_seeds_sub(args):
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
    args[0].tracking_field.dataset.data = np.load(data_file_info[0],
                                                  mmap_mode=data_file_info[1])

    try:
        streamlines, seeds = get_streamlines_at_seeds(*args)
        return streamlines, seeds
    except Exception as e:
        logging.error("Operation _get_streamlines_sub() failed.")
        traceback.print_exception(*sys.exc_info(), file=sys.stderr)
        raise e


def get_streamlines_at_seeds(tracker, mask, seed_generator, chunk_id, params,
                             compression_th=0.1, nbr_processes=1,
                             save_seeds=True):
    """
    Generate streamlines from all initial positions following the tracking
    parameters.

    Parameters
    ----------
    tracker : AbstractTracker
        Tracking object.
    mask : BinaryMask
        Tracking volume(s).
    seed_generator : SeedGenerator
        Seeding volume.
    chunk_id: int
        This chunk id.
    params: TrackingParams
        Tracking parameters, see scilpy.tracking.utils.py.
    compression_th : float,
        Maximal distance threshold for compression. If None or 0, no
        compression is applied.
    nbr_processes: int
        Number of sub processes to use.
    save_seeds: bool
        Whether to save the seeds associated to their respective streamlines.


    Returns
    -------
    lines: list, list of list of 3D positions
    """

    streamlines = []
    seeds = []
    # Initialize the random number generator to cover multiprocessing, skip,
    # which voxel to seed and the subvoxel random position
    chunk_size = int(params.nbr_seeds / nbr_processes)
    skip = params.skip

    first_seed_of_chunk = chunk_id * chunk_size + skip
    random_generator, indices = \
        seed_generator.init_pos(params.random, first_seed_of_chunk)

    if chunk_id == nbr_processes - 1:
        chunk_size += params.nbr_seeds % nbr_processes
    for s in range(chunk_size):
        if s % 1000 == 0:
            logging.info(str(os.getpid()) + " : " + str(s)
                         + " / " + str(chunk_size))

        seed = \
            seed_generator.get_next_pos(random_generator, indices,
                                        first_seed_of_chunk + s)
        line = get_line_from_seed(tracker, mask, seed, params)
        if line is not None:
            if compression_th and compression_th > 0:
                streamlines.append(
                    compress_streamlines(np.array(line, dtype='float32'),
                                         compression_th))
            else:
                streamlines.append((np.array(line, dtype='float32')))

            if save_seeds:
                seeds.append(np.asarray(seed, dtype='float32'))

    return streamlines, seeds


def get_line_from_seed(tracker: AbstractTracker, mask: AccessibleVolume, pos,
                       params):
    """
    Generate a streamline from an initial position following the tracking
    parameters.

    Parameters
    ----------
    tracker : AbstractTracker
        Tracking object.
    mask : BinaryMask
        Tracking volume(s).
    pos : tuple
        3D position, the seed position.
    params: TrackingParams
        Tracking parameters.

    Returns
    -------
    line: list of 3D positions
    """
    np.random.seed(np.uint32(hash((pos, params.random))))
    line = []
    if tracker.initialize(pos):
        forward = _get_line(tracker, mask, params, True)
        if forward is not None and len(forward) > 0:
            line.extend(forward)

        if not params.is_single_direction and forward is not None:
            backward = _get_line(tracker, mask, params, False)
            if backward is not None and len(backward) > 0:
                line.reverse()
                line.pop()
                line.extend(backward)
        else:
            backward = []

        if ((len(line) > 1 and
             forward is not None and
             backward is not None and
             params.min_nbr_pts <= len(line) <= params.max_nbr_pts)):
            return line
        elif params.is_keep_single_pts and params.min_nbr_pts == 1:
            return [pos]
        return None
    if ((params.is_keep_single_pts and
         params.min_nbr_pts == 1)):
        return [pos]
    return None


def _get_line(tracker, mask, params, is_forward):
    line = _get_line_binary(tracker, mask, params, is_forward)

    while (line is not None and len(line) > 0 and
           not tracker.is_position_in_bound(line[-1])):
        line.pop()

    return line


def _get_line_binary(tracker, mask, params, is_forward):
    """
    This function is use for binary mask.
    Generate a streamline in forward or backward direction from an initial
    position following the tracking parameters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    params: TrackingParams, tracking parameters.
    is_forward: bool, track in forward direction if True,
                      track in backward direction if False.

    Returns
    -------
    line: list of 3D positions
    """
    line = [tracker.init_pos]
    line_dirs = [tracker.forward_dir] if is_forward else [tracker.backward_dir]

    no_valid_direction_count = 0

    propagation_can_continue = True
    while len(line) < params.max_nbr_pts and propagation_can_continue:
        new_pos, new_dir, is_valid_direction = tracker.propagate(
            line[-1], line_dirs[-1])
        line.append(new_pos)
        line_dirs.append(new_dir)

        if is_valid_direction:
            no_valid_direction_count = 0
        else:
            no_valid_direction_count += 1

        if no_valid_direction_count > params.max_no_dir:
            return line

        propagation_can_continue = (mask.get_position_value(*line[-1]) > 0 and
                                    mask.is_position_in_bound(*line[-1]))

    # make a last step in the last direction
    line.append(line[-1] + tracker.step_size * np.array(line_dirs[-1]))
    return line
