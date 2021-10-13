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


data_file_info = None


def track(tracker, mask, seed, param, compress=False,
          compression_th=0.1, nbr_processes=1, save_seeds=False):
    """
    Generate a set of streamline from seed, mask and odf files.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seed : Seed, seeding volume.
    param: TrackingParams,
        tracking parameters, see scilpy.tracking.utils.py.
    compress : bool, enable streamlines compression.
    compression_th : float,
        maximal distance threshold for compression.
    nbr_processes: int, number of sub processes to use.
    save_seeds: bool, whether to save the seeds associated to their
        respective streamlines
    Return
    ------
    streamlines: list of numpy.array
    seeds: list of numpy.array
    """
    if param.nbr_streamlines == 0:
        if nbr_processes <= 0:
            try:
                nbr_processes = multiprocessing.cpu_count()
            except NotImplementedError:
                warnings.warn("Cannot determine number of cpus. \
                    returns nbr_processes set to 1.")
                nbr_processes = 1

        param.processes = nbr_processes
        if param.processes > param.nbr_seeds:
            nbr_processes = param.nbr_seeds
            param.processes = param.nbr_seeds
            logging.debug('Setting number of processes to ' +
                          str(param.processes) +
                          ' since there were less seeds than processes.')
        chunk_id = np.arange(nbr_processes)
        if nbr_processes < 2:
            lines, seeds = get_streamlines(tracker, mask, seed, chunk_id,
                                           param, compress, compression_th,
                                           save_seeds=save_seeds)
        else:

            with nib.tmpdirs.InTemporaryDirectory() as tmpdir:

                # must be better designed for dipy
                # the tracking should not know which data to deal with
                data_file_name = os.path.join(tmpdir, 'data.npy')
                np.save(data_file_name, tracker.tracking_field.dataset.data)
                tracker.tracking_field.dataset.data = None

                pool = multiprocessing.Pool(nbr_processes,
                                            initializer=_init_sub_process,
                                            initargs=(data_file_name,
                                                      param.mmap_mode))

                max_tries = 100  # default value for max_tries
                lines_per_process, seeds_per_process = zip(*pool.map(
                    _get_streamlines_sub, zip(itertools.repeat(tracker),
                                              itertools.repeat(mask),
                                              itertools.repeat(seed),
                                              chunk_id,
                                              itertools.repeat(param),
                                              itertools.repeat(compress),
                                              itertools.repeat(compression_th),
                                              itertools.repeat(max_tries),
                                              itertools.repeat(save_seeds))))
                pool.close()
                # Make sure all worker processes have exited before leaving
                # context manager in order to prevent temporary file deletion
                # errors in Windows
                pool.join()
                lines =\
                    np.array([line for line in
                              itertools.chain(*lines_per_process)])
                seeds =\
                    np.array([seed for seed in
                              itertools.chain(*seeds_per_process)])
    else:
        if nbr_processes > 1:
            warnings.warn("No multiprocessing implemented while computing " +
                          "a fixed number of streamlines.")
        lines, seeds = get_n_streamlines(tracker, mask, seed, param, compress,
                                         compression_th, save_seeds=save_seeds)

    return lines, seeds


def _init_sub_process(date_file_name, mmap_mod):
    global data_file_info
    data_file_info = (date_file_name, mmap_mod)
    return


def _get_streamlines_sub(args):
    """
    multiprocessing.pool.map input function.

    Parameters
    ----------
    args : List, parameters for the get_lines(*) function.

    Return
    -------
    lines: list, list of list of 3D positions (streamlines).
    """
    global data_file_info
    args[0].tracking_field.dataset.data = np.load(data_file_info[0],
                                                  mmap_mode=data_file_info[1])

    try:
        streamlines, seeds = get_streamlines(*args[0:9])
        return streamlines, seeds
    except Exception as e:
        logging.error("Operation _get_streamlines_sub() failed.")
        traceback.print_exception(*sys.exc_info(), file=sys.stderr)
        raise e


def get_n_streamlines(tracker, mask, seeding_mask, param,
                      compress=False, compression_error_threshold=0.1,
                      max_tries=100, save_seeds=True):
    """
    Generate N valid streamlines

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seeding_mask : Seed, seeding volume.
    param: TrackingParams, tracking parameters.
    compress : bool, enable streamlines compression.
    compression_error_threshold : float,
        maximal distance threshold for compression.
    max_tries: int
    save_seeds: bool

    Returns
    -------
    lines: list, list of list of 3D positions (streamlines)
    """

    i = 0
    streamlines = []
    seeds = []
    skip = 0
    # Initialize the random number generator, skip,
    # which voxel to seed and the subvoxel random position
    first_seed_of_chunk = np.int32(param.skip)
    random_generator, indices =\
        seeding_mask.init_pos(param.random, first_seed_of_chunk)
    while (len(streamlines) < param.nbr_streamlines and
           skip < param.nbr_streamlines * max_tries):
        if i % 1000 == 0:
            logging.info(str(os.getpid()) + " : " +
                         str(len(streamlines)) + " / " +
                         str(param.nbr_streamlines))
        seed = seeding_mask.get_next_pos(random_generator,
                                         indices,
                                         first_seed_of_chunk + i)
        line = get_line_from_seed(tracker, mask, seed, param)
        if line is not None:
            if compress:
                streamlines.append(
                    compress_streamlines(np.array(line, dtype='float32'),
                                         compression_error_threshold))
            else:
                streamlines.append((np.array(line, dtype='float32')))
            if save_seeds:
                seeds.append(np.asarray(seed, dtype='float32'))

        i += 1
    return streamlines, seeds


def get_streamlines(tracker, mask, seeding_mask, chunk_id, param,
                    compress=False, compression_error_threshold=0.1,
                    save_seeds=True):
    """
    Generate streamlines from all initial positions
    following the tracking parameters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seeding_mask : Seed, seeding volume.
    chunk_id: int, chunk id.
    param: TrackingParams, tracking parameters.
    compress : bool, enable streamlines compression.
    compression_error_threshold : float,
        maximal distance threshold for compression.
    save_seeds: bool

    Returns
    -------
    lines: list, list of list of 3D positions
    """

    streamlines = []
    seeds = []
    # Initialize the random number generator to cover multiprocessing, skip,
    # which voxel to seed and the subvoxel random position
    chunk_size = int(param.nbr_seeds / param.processes)
    skip = param.skip

    first_seed_of_chunk = chunk_id * chunk_size + skip
    random_generator, indices =\
        seeding_mask.init_pos(param.random,
                              first_seed_of_chunk)

    if chunk_id == param.processes - 1:
        chunk_size += param.nbr_seeds % param.processes
    for s in range(chunk_size):
        if s % 1000 == 0:
            logging.info(str(os.getpid()) + " : " + str(s)
                         + " / " + str(chunk_size))

        seed =\
            seeding_mask.get_next_pos(random_generator,
                                      indices,
                                      first_seed_of_chunk + s)
        line = get_line_from_seed(tracker, mask, seed, param)
        if line is not None:
            if compress:
                streamlines.append(
                    compress_streamlines(np.array(line, dtype='float32'),
                                         compression_error_threshold))
            else:
                streamlines.append((np.array(line, dtype='float32')))

            if save_seeds:
                seeds.append(np.asarray(seed, dtype='float32'))

    return streamlines, seeds


def get_line_from_seed(tracker, mask, pos, param):
    """
    Generate a streamline from an initial position following the tracking
    parameters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    pos : tuple, 3D position, the seed position.
    param: TrackingParams, tracking parameters.

    Returns
    -------
    line: list of 3D positions
    """

    np.random.seed(np.uint32(hash((pos, param.random))))
    line = []
    if tracker.initialize(pos):
        forward = _get_line(tracker, mask, param, True)
        if forward is not None and len(forward) > 0:
            line.extend(forward)

        if not param.is_single_direction and forward is not None:
            backward = _get_line(tracker, mask, param, False)
            if backward is not None and len(backward) > 0:
                line.reverse()
                line.pop()
                line.extend(backward)
        else:
            backward = []

        if ((len(line) > 1 and
             forward is not None and
             backward is not None and
             param.min_nbr_pts <= len(line) <= param.max_nbr_pts)):
            return line
        elif param.is_keep_single_pts and param.min_nbr_pts == 1:
            return [pos]
        return None
    if ((param.is_keep_single_pts and
         param.min_nbr_pts == 1)):
        return [pos]
    return None


def _get_line(tracker, mask, param, is_forward):
    line = _get_line_binary(tracker, mask, param, is_forward)

    while (line is not None and len(line) > 0 and
           not tracker.is_position_in_bound(line[-1])):
        line.pop()

    return line


def _get_line_binary(tracker, mask, param, is_forward):
    """
    This function is use for binary mask.
    Generate a streamline in forward or backward direction from an initial
    position following the tracking parameters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    param: TrackingParams, tracking parameters.
    is_forward: bool, track in forward direction if True,
                      track in backward direction if False.

    Returns
    -------
    line: list of 3D positions
    """
    line = [tracker.init_pos]
    line_dirs = [tracker.forward_dir] if is_forward else [tracker.backward_dir]

    no_valid_direction_count = 0

    while (len(line) < param.max_nbr_pts and
           mask.isPropagationContinues(line[-1])):
        new_pos, new_dir, is_valid_direction = tracker.propagate(
            line[-1], line_dirs[-1])
        line.append(new_pos)
        line_dirs.append(new_dir)

        if is_valid_direction:
            no_valid_direction_count = 0
        else:
            no_valid_direction_count += 1

        if no_valid_direction_count > param.max_no_dir:
            return line

    # make a last step in the last direction
    line.append(line[-1] + tracker.step_size * np.array(line_dirs[-1]))
    return line
