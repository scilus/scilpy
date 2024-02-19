# -*- coding: utf-8 -*-
import numpy as np


def project_map_to_streamlines(sft, map_volume, endpoints_only=False):
    """
    Projects a map onto the points of streamlines.

    Parameters
    ----------
    sft: StatefulTractogram
        Input tractogram.
    map_volume: DataVolume
        Input map.

    Optional:
    ---------
    endpoints_only: bool
        If True, will only project the map_volume onto the endpoints of the
        streamlines (all values along streamlines set to zero). If False,
        will project the map_volume onto all points of the streamlines.

    Returns
    -------
    streamline_data:
        map_volume projected to each point of the streamlines.
    """
    if len(map_volume.data.shape) == 4:
        dimension = map_volume.data.shape[3]
    else:
        dimension = 1

    streamline_data = []
    if endpoints_only:
        for s in sft.streamlines:
            p1_data = map_volume.get_value_at_coordinate(
                s[0][0], s[0][1], s[0][2],
                space=sft.space, origin=sft.origin)
            p2_data = map_volume.get_value_at_coordinate(
                s[-1][0], s[-1][1], s[-1][2],
                space=sft.space, origin=sft.origin)

            thisstreamline_data = np.ones((len(s), dimension)) * np.nan

            thisstreamline_data[0] = p1_data
            thisstreamline_data[-1] = p2_data
            thisstreamline_data = np.asarray(thisstreamline_data)

            streamline_data.append(
                np.reshape(thisstreamline_data,
                           (len(thisstreamline_data), dimension)))
    else:
        for s in sft.streamlines:
            thisstreamline_data = []
            for p in s:
                thisstreamline_data.append(map_volume.get_value_at_coordinate(
                    p[0], p[1], p[2], space=sft.space, origin=sft.origin))

            streamline_data.append(
                np.reshape(thisstreamline_data,
                           (len(thisstreamline_data), dimension)))

    return streamline_data
