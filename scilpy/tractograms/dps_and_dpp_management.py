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


def perform_streamline_operation_per_point(op_name, sft, dpp_name='metric',
                                           endpoints_only=False):
    """Peforms an operation per point for all streamlines.

    Parameters
    ----------
    op_name: str
        A callable that takes a list of streamline data per point (4D) and
        returns a list of streamline data per point.
    sft: StatefulTractogram
        The streamlines used in the operation.
    dpp_name: str
        The name of the data per point to be used in the operation.
    endpoints_only: bool
        If True, will only perform operation on endpoints

    Returns
    -------
    new_sft: StatefulTractogram
        sft with data per streamline resulting from the operation.
    """

    # Performing operation
    call_op = OPERATIONS[op_name]
    if endpoints_only:
        new_data_per_point = []
        for s in sft.data_per_point[dpp_name]:
            this_data_per_point = np.nan * np.ones((len(s), 1))
            this_data_per_point[0] = call_op(s[0])
            this_data_per_point[-1] = call_op(s[-1])
            new_data_per_point.append(
                np.reshape(this_data_per_point, (len(this_data_per_point), 1)))
    else:
        new_data_per_point = []
        for s in sft.data_per_point[dpp_name]:
            this_data_per_point = []
            for p in s:
                this_data_per_point.append(call_op(p))
            new_data_per_point.append(
                np.reshape(this_data_per_point, (len(this_data_per_point), 1)))

    # Extracting streamlines
    return new_data_per_point


def perform_operation_per_streamline(op_name, sft, dpp_name='metric',
                                     endpoints_only=False):
    """Performs an operation across all data points for each streamline.

    Parameters
    ----------
    op_name: str
        A callable that takes a list of streamline data per streamline and
        returns a list of data per streamline.
    sft: StatefulTractogram
        The streamlines used in the operation.
    dpp_name: str
        The name of the data per point to be used in the operation.
    endpoints_only: bool
        If True, will only perform operation on endpoints

    Returns
    -------
    new_sft: StatefulTractogram
        sft with data per streamline resulting from the operation.
    """
    # Performing operation
    call_op = OPERATIONS[op_name]
    if endpoints_only:
        new_data_per_streamline = []
        for s in sft.data_per_point[dpp_name]:
            start = s[0]
            end = s[-1]
            concat = np.concatenate((start[:], end[:]))
            new_data_per_streamline.append(call_op(concat))
    else:
        new_data_per_streamline = []
        for s in sft.data_per_point[dpp_name]:
            s_np = np.asarray(s)
            new_data_per_streamline.append(call_op(s_np))

    return new_data_per_streamline


def perform_pairwise_streamline_operation_on_endpoints(op_name, sft,
                                                       dpp_name='metric'):
    """Peforms an operation across endpoints for each streamline.

    Parameters
    ----------
    op_name: str
        A callable that takes a list of streamline data per streamline and
        returns a list of data per streamline.
    sft: StatefulTractogram
        The streamlines used in the operation.
    dpp_name: str
        The name of the data per point to be used in the operation.

    Returns
    -------
    new_sft: StatefulTractogram
        sft with data per streamline resulting from the operation.
    """
    # Performing operation
    call_op = OPERATIONS[op_name]
    new_data_per_streamline = []
    for s in sft.data_per_point[dpp_name]:
        new_data_per_streamline.append(call_op(s[0], s[-1])[0, 1])

    return new_data_per_streamline


def stream_mean(array):
    return np.squeeze(np.mean(array, axis=0))


def stream_sum(array):
    return np.squeeze(np.sum(array, axis=0))


def stream_min(array):
    return np.squeeze(np.min(array, axis=0))


def stream_max(array):
    return np.squeeze(np.max(array, axis=0))


def stream_correlation(array1, array2):
    return np.corrcoef(array1, array2)


OPERATIONS = {
    'mean': stream_mean,
    'sum': stream_sum,
    'min': stream_min,
    'max': stream_max,
    'correlation': stream_correlation,
}
