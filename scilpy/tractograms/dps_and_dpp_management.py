# -*- coding: utf-8 -*-
import numpy as np


def project_map_to_streamlines(sft, map, endpoints_only=False):
    """
    Projects a map onto the points of streamlines.

    Parameters
    ----------
    sft: StatefulTractogram
        Input tractogram.
    map: DataVolume
        Input map.

    Optional:
    ---------
    endpoints_only: bool
        If True, will only project the map onto the endpoints of the
        streamlines (all values along streamlines set to zero). If False,
        will project the map onto all points of the streamlines.

    Returns
    -------
    streamline_data:
        map projected to each point of the streamlines.
    """
    if len(map.data.shape) == 4:
        dimension = map.data.shape[3]
    else:
        dimension = 1

    streamline_data = []
    if endpoints_only:
        for s in sft.streamlines:
            p1_data = map.get_value_at_coordinate(
                s[0][0], s[0][1], s[0][2],
                space=sft.space, origin=sft.origin)
            p2_data = map.get_value_at_coordinate(
                s[-1][0], s[-1][1], s[-1][2],
                space=sft.space, origin=sft.origin)
            thisstreamline_data = []
            if dimension == 1:
                thisstreamline_data = np.ones((len(s), 1)) * np.nan
            else:
                thisstreamline_data = np.ones(
                    (len(s), p1_data.shape[0])) * np.nan

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
                thisstreamline_data.append(map.get_value_at_coordinate(
                    p[0], p[1], p[2], space=sft.space, origin=sft.origin))

            streamline_data.append(
                np.reshape(thisstreamline_data,
                           (len(thisstreamline_data), dimension)))

    return streamline_data
