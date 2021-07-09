# -*- coding: utf-8 -*-
import itertools
import logging
import multiprocessing
import os
import sys
import traceback
import warnings
import json

import nibabel as nib
import numpy as np

from dipy.tracking.streamlinespeed import compress_streamlines


data_file_info = None


def track(tracker, mask, seed, param, resampled_tracker=None, region_mr=None,
          resampled_mask=None, compress=False, compression_th=0.1,
          nbr_processes=1, pft_tracker=None, save_seeds=False):
    """
    Generate a set of streamline from seed, mask and odf files.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seed : Seed, seeding volume.
    param: TrackingParams,
        tracking parameters, see scilpy.tracking.utils.py.
    tracker_mr : Tracker, tracking object at lower resolution
    region_mr : Mask, to limit the multiresolution tracking region
    mask_mr : Mask, resampled tracking volume(s)
    compress : bool, enable streamlines compression.
    compression_th : float,
        maximal distance threshold for compression.
    nbr_processes: int, number of sub processes to use.
    pft_tracker: Tracker, tracking object for pft module.
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
            lines, seeds = get_streamlines(
                tracker, mask, seed, chunk_id, pft_tracker, param,
                compress=compress, compression_error_threshold=compression_th,
                resampled_tracker=resampled_tracker, region_mr=region_mr,
                resampled_mask=resampled_mask, save_seeds=save_seeds)
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
                                              itertools.repeat(pft_tracker),
                                              itertools.repeat(param),
                                              itertools.repeat(resampled_tracker),
                                              itertools.repeat(resampled_mask),
                                              itertools.repeat(region_mr),
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
        lines, seeds = get_n_streamlines(tracker, mask, seed,
                                         pft_tracker, param,
                                         compress,
                                         compression_th,
                                         save_seeds=save_seeds)

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
        streamlines, seeds = get_streamlines(*args[0:12])
        return streamlines, seeds
    except Exception as e:
        print("error")
        traceback.print_exception(*sys.exc_info(), file=sys.stderr)
        raise e


def get_n_streamlines(tracker, mask, seeding_mask, pft_tracker, param,
                      compress=False, compression_error_threshold=0.1,
                      max_tries=100, save_seeds=True):
    """
    Generate N valid streamlines

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seeding_mask : Seed, seeding volume.
    pft_tracker: Tracker, tracking object for pft module.
    param: TrackingParams, tracking parameters.
    compress : bool, enable streamlines compression.
    compression_error_threshold : float,
        maximal distance threshold for compression.

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
            print(str(os.getpid()) + " : " +
                  str(len(streamlines)) + " / " +
                  str(param.nbr_streamlines))
        seed = seeding_mask.get_next_pos(random_generator,
                                         indices,
                                         first_seed_of_chunk + i)
        line = get_line_from_seed(tracker, mask, seed,
                                  pft_tracker, param)
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


def get_streamlines(tracker, mask, seeding_mask, chunk_id, pft_tracker, param,
                    resampled_tracker=None, region_mr=None,
                    resampled_mask=None, compress=False,
                    compression_error_threshold=0.1, save_seeds=True,):
    """
    Generate streamlines from all initial positions
    following the tracking parameters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    seeding_mask : Seed, seeding volume.
    chunk_id: int, chunk id.
    pft_tracker: Tracker, tracking object for pft module.
    param: TrackingParams, tracking parameters.
    compress : bool, enable streamlines compression.
    compression_error_threshold : float,
        maximal distance threshold for compression.

    Returns
    -------
    lines: list, list of list of 3D positions
    """
    streamlines = []
    seeds = []
    lines = 0
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
            print(str(os.getpid()) + " : " + str(
                s) + " / " + str(chunk_size))

        seed =\
            seeding_mask.get_next_pos(random_generator,
                                      indices,
                                      first_seed_of_chunk + s)

        line = get_line_from_seed(tracker, mask, seed, pft_tracker, param,
                                  resampled_tracker, region_mr, resampled_mask)

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


def get_line_from_seed(tracker, mask, pos, pft_tracker, param,
                       resampled_tracker=None, region_mr=None,
                       resampled_mask=None):
    """
    Generate a streamline from an initial position following the tracking
    paramters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    pos : tuple, 3D position, the seed position.
    pft_tracker: Tracker, tracking object for pft module.
    param: TrackingParams, tracking parameters.

    Returns
    -------
    line: list of 3D positions
    """

    np.random.seed(np.uint32(hash((pos, param.random))))

    line = []
    if tracker.initialize(pos):
        forward = _get_line(tracker, mask, pft_tracker, param, True,
                            resampled_tracker, region_mr, resampled_mask)
        if forward is not None and len(forward) > 0:
            line.extend(forward)

        if not param.is_single_direction and forward is not None:
            backward = _get_line(tracker, mask, pft_tracker, param, False,
                                 resampled_tracker, region_mr, resampled_mask)
            if backward is not None and len(backward) > 0:
                line.reverse()
                line.pop()
                line.extend(backward)
        else:
            backward = []

        if ((len(line) > 1 and
             forward is not None and
             backward is not None and
             len(line) >= param.min_nbr_pts and
             len(line) <= param.max_nbr_pts)):
            return line
        elif (param.is_keep_single_pts and
              param.min_nbr_pts == 1):
            return [pos]
        return None
    if ((param.is_keep_single_pts and
         param.min_nbr_pts == 1)):
        return [pos]
    return None


def _get_line(tracker, mask, pft_tracker, param, is_forward,
              resampled_tracker=None, region_mr=None, resampled_mask=None):
    line = _get_line_binary(tracker, mask, param, is_forward,
                            resampled_tracker, region_mr, resampled_mask)

    while (line is not None and len(line) > 0 and
           not tracker.isPositionInBound(line[-1])):
        line.pop()

    return line


def _get_line_binary(tracker, mask, param, is_forward, resampled_tracker=None,
                     region_mr=None, resampled_mask=None):
    """
    This function is use for binary mask.
    Generate a streamline in forward or backward direction from an initial
    position following the tracking paramters.

    Parameters
    ----------
    tracker : Tracker, tracking object.
    mask : Mask, tracking volume(s).
    param: TrackingParams, tracking parameters.
    is_forward: bool, track in forward direction if True,
                      track in backward direction if False.
    resampled_tracker : Tracker at lower resolution, tracking object.
    region_mr : Mask of hard-to-track regions, used for multiresolution
    resampled_mask : Mask at lower resolution, tracking volume(s).

    Returns
    -------
    line: list of 3D positions
    """
    # Version 1
    line = [tracker.init_pos]

    line_dirs = [tracker.forward_dir] if is_forward else [tracker.backward_dir]
    no_valid_direction_count = 0

    while (len(line) < param.max_nbr_pts and
           mask.isPropagationContinues(line[-1])):

        if param.is_mr:
            # If position is in region of interest for multiresolution
            # Use Tracker at lower resolution to get new position and
            # new direction
            if (region_mr.isPropagationContinues(line[-1]) and
                    mask.isPropagationContinues(line[-1])):
                new_pos, new_dir, is_valid_direction =\
                    resampled_tracker.propagate(line[-1], line_dirs[-1])
                # print('is_looking_for_lower_direction')
            else:
                new_pos, new_dir, is_valid_direction = tracker.propagate(
                    line[-1], line_dirs[-1])
                # print('tried but is not in mask')
        else:
            new_pos, new_dir, is_valid_direction = tracker.propagate(
                line[-1], line_dirs[-1])
            # print('was not in mr mode')

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

    # Version 2
    # line = [tracker.init_pos]

    # line_dirs = [tracker.forward_dir] if is_forward else [tracker.backward_dir]
    # no_valid_direction_count = 0
    # cpt = 0
    # while (len(line) < param.max_nbr_pts and
    #        mask.isPropagationContinues(line[-1])):
    #     new_pos, new_dir, is_valid_direction = tracker.propagate(
    #         line[-1], line_dirs[-1])

    #     line.append(new_pos)
    #     line_dirs.append(new_dir)

    #     if is_valid_direction:
    #         no_valid_direction_count = 0
    #     else:
    #         no_valid_direction_count += 1

    #     if no_valid_direction_count > param.max_no_dir:
    #         if param.is_mr:
    #             # If there is no valid direction, but is a hard-to-track region
    #             # Repeat same process, but in lower resolution
    #             if (region_mr.isPropagationContinues(line[-1]) and
    #                     resampled_mask.isPropagationContinues(line[-1])):
    #                 no_valid_direction_count = 0
    #                 # print('restart with lower res')
    #                 while (len(line) < param.max_nbr_pts and
    #                         mask.isPropagationContinues(line[-1])):
    #                     new_pos, new_dir, is_valid_direction =\
    #                         resampled_tracker.propagate(line[-1],
    #                                                     line_dirs[-1])
    #                     # print('cherche direction en lower res')
    #                     line.append(new_pos)
    #                     line_dirs.append(new_dir)

    #                     if is_valid_direction:
    #                         no_valid_direction_count = 0
    #                     else:
    #                         no_valid_direction_count += 1

    #                     if no_valid_direction_count > param.max_no_dir:
    #                         return line
    #             else:
    #                 return line
    #         else:
    #             return line

    # # make a last step in the last direction
    # line.append(line[-1] + tracker.step_size * np.array(line_dirs[-1]))
    # return line

    # Version 3
    # line = [tracker.init_pos]

    # line_dirs = [tracker.forward_dir] if is_forward else [tracker.backward_dir]
    # no_valid_direction_count = 0

    # while (len(line) < param.max_nbr_pts and
    #        mask.isPropagationContinues(line[-1])):

    #     new_pos, new_dir, is_valid_direction = tracker.propagate(
    #         line[-1], line_dirs[-1])

    #     # If the direction is not valid in this resolution
    #     if not is_valid_direction:
    #         if param.is_mr:
    #             # If the position is in the region of interest for
    #             # multiresolution. Use Tracker at lower resolution to get the
    #             # new position and direction
    #             if (region_mr.isPropagationContinues(line[-1]) and
    #                     resampled_mask.isPropagationContinues(line[-1])):
    #                 new_pos, new_dir, is_valid_direction =\
    #                     resampled_tracker.propagate(line[-1], line_dirs[-1])
    #                 print('cherche nouvelle dir en lower res')
    #                 print(no_valid_direction_count)

    #     line.append(new_pos)
    #     line_dirs.append(new_dir)

    #     if is_valid_direction:
    #         no_valid_direction_count = 0
    #     else:
    #         no_valid_direction_count += 1

    #     if no_valid_direction_count > param.max_no_dir:
    #         return line
    # # make a last step in the last direction
    # line.append(line[-1] + tracker.step_size * np.array(line_dirs[-1]))
    # return line

