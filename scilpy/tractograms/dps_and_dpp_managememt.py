# -*- coding: utf-8 -*-
import logging

import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram


def average_dpp_as_dps(sft: StatefulTractogram, dpp_keys, remove_dpp=True,
                       endpoints_only=False):
    """
    Parameters
    ----------
    sft: StatefulTractogram
    dpp_keys: list
        The dpp keys to transform to dps,
    remove_dpp: bool
        If true, remove the transformed dpp from data.
    endpoints_only: bool
        If true, each line's dps will the the mean of its two endpoints.

    Returns
    -------
    sft: StatefulTractogram
        The modified sft. The dpp key is added as a dps key.
    """
    for key in dpp_keys:
        assert key not in sft.data_per_streamline.keys(), \
            "The dpp key already exists as a dps key. Can't convert."

        dpp = sft.data_per_point[key]

        if remove_dpp:
            sft.data_per_point[key] = []

        if endpoints_only:
            sft.data_per_streamline[key] = [np.mean(s[[0, -1], :])
                                            for s in dpp]
        else:
            sft.data_per_streamline[key] = [np.mean(s) for s in dpp]

    return sft


def repeat_dps_as_dpp(sft: StatefulTractogram, dps_keys, remove_dps=True):
    """
    Parameters
    ----------
    sft: StatefulTractogram
    dps_keys: list
        The dps keys to transform to dpp,
    remove_dps: bool
        If true, remove the transformed dps from data.

    Returns
    -------
    sft: StatefulTractogram
        The modified sft. The dps key is added as a dpp key.
    """
    for key in dps_keys:
        assert key not in sft.data_per_point.keys(), \
            "The dpp key already exists as a dps key. Can't convert."

        dps = sft.data_per_streamline[key]

        if remove_dps:
            sft.data_per_streamline[key] = []

        sft.data_per_point[key] = [val*np.ones(len(s))
                                   for val, s in zip(dps, sft.streamlines)]

    return sft


def keep_only_endpoints(sft):
    """
    Keeps only the endpoints, both in the streamlines and in associated
    data_per_point.
    """
    sft.streamlines = [s[[0, -1], :] for s in sft.streamlines]
    for key in sft.data_per_point.keys():
        sft.data_per_point[key] = [s[[0, -1]] for s in sft.data_per_point[key]]
    return sft


def project_dpp_to_map(sft, dpp_keys, sum_lines=False):
    """
    Saves the values of data_per_point keys to the underlying voxels. Averages
    the values of various streamlines in each voxel. Returns one map per key.

    The streamlines are not preprocessed here. You should probably first
    uncompress your streamlines to have smoother maps.

    Parameters
    ----------
    sft: StatefulTractogram
    dpp_keys: list
    sum_lines: bool
        Do not average streamlines; sum them instead.

    Returns
    -------
    maps: list
    """
    sft.to_vox()
    sft.to_corner()
    epsilon = 1e-6

    count = np.zeros(sft.dimensions)
    maps = []
    for i, dpp_key in enumerate(dpp_keys):
        logging.info("Projecting streamlines metric {} to a map"
                     .format(dpp_key))
        current_map = np.zeros(sft.dimensions)
        for s in range(len(sft)):
            for p in range(len(sft.streamlines[s])):
                x, y, z = sft.streamlines[s][p, :].astype(int)
                if i == 0:
                    # Only counting for the first key
                    count[x, y, z] += 1

                current_map[x, y, z] += sft.data_per_point[dpp_key][s][p]

        maps.append(current_map)
        if not sum_lines:
            if i == 0:
                count = np.maximum(count, epsilon)
            maps[i] /= count

    return maps
