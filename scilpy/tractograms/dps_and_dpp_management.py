# -*- coding: utf-8 -*-
import numpy as np

from scilpy.viz.color import clip_and_normalize_data_for_cmap


def add_data_as_color_dpp(sft, cmap, data, clip_outliers=False, min_range=None,
                          max_range=None, min_cmap=None, max_cmap=None,
                          log=False, LUT=None):
    """
    Normalizes data between 0 and 1 for an easier management with colormaps.
    The real lower bound and upperbound are returned.

    Data can be clipped to (min_range, max_range) before normalization.
    Alternatively, data can be kept as is, but the colormap be fixed to
    (min_cmap, max_cmap).

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram
    cmap: plt colormap
        The colormap. Ex, see scilpy.viz.utils.get_colormap().
    data: np.ndarray or list[list] or list[np.ndarray]
        The data to convert to color. Expecting one value per point to add as
        dpp. If instead data has one value per streamline, setting the same
        color to all points of the streamline (as dpp).
        Either a vector numpy array (all streamlines concatenated), or a list
        of arrays per streamline.
    clip_outliers: bool
        See description of the following parameters in
        clip_and_normalize_data_for_cmap.
    min_range: float
        Data values below min_range will be clipped.
    max_range: float
        Data values above max_range will be clipped.
    min_cmap: float
        Minimum value of the colormap. Most useful when min_range and max_range
        are not set; to fix the colormap range without modifying the data.
    max_cmap: float
        Maximum value of the colormap. Idem.
    log: bool
        If True, apply a logarithmic scale to the data.
    LUT: np.ndarray
        If set, replaces the data values by the Look-Up Table values. In order,
        the first value of the LUT is set everywhere where data==1, etc.

    Returns
    -------
    sft: StatefulTractogram
        The tractogram, with dpp 'color' added.
    lbound: float
        The lower bound of the associated colormap.
    ubound: float
        The upper bound of the associated colormap.
    """
    # If data is a list of lists, merge.
    if isinstance(data[0], list) or isinstance(data[0], np.ndarray):
        data = np.hstack(data)

    values, lbound, ubound = clip_and_normalize_data_for_cmap(
        data, clip_outliers, min_range, max_range,
        min_cmap, max_cmap, log, LUT)

    # Important: values are in float after clip_and_normalize.
    color = np.asarray(cmap(values)[:, 0:3]) * 255
    if len(color) == len(sft):
        tmp = [np.tile([color[i][0], color[i][1], color[i][2]],
                       (len(sft.streamlines[i]), 1))
               for i in range(len(sft.streamlines))]
        sft.data_per_point['color'] = tmp
    elif len(color) == len(sft.streamlines._data):
        sft.data_per_point['color'] = sft.streamlines
        sft.data_per_point['color']._data = color
    else:
        raise ValueError("Error in the code... Colors do not have the right "
                         "shape. Expecting either one color per streamline "
                         "({}) or one per point ({}) but got {}."
                         .format(len(sft), len(sft.streamlines._data),
                                 len(color)))
    return sft, lbound, ubound


def convert_dps_to_dpp(sft, keys, overwrite=False):
    """
    Copy the value of the data_per_streamline to each point of the
    streamline, as data_per_point. The dps key is removed and added as dpp key.

    Parameters
    ----------
    sft: StatefulTractogram
    keys: str or List[str], optional
        The list of dps keys to convert to dpp.
    overwrite: bool
        If true, allow continuing even if the key already existed as dpp.
    """
    if isinstance(keys, str):
        keys = [keys]

    for key in keys:
        if key not in sft.data_per_streamline:
            raise ValueError(
                "Dps key {} not found! Existing dps keys: {}"
                .format(key, list(sft.data_per_streamline.keys())))
        if key in sft.data_per_point and not overwrite:
            raise ValueError("Dpp key {} already existed. Please allow "
                             "overwriting.".format(key))
        sft.data_per_point[key] = [[val]*len(s) for val, s in
                                   zip(sft.data_per_streamline[key],
                                       sft.streamlines)]
        del sft.data_per_streamline[key]

    return sft


def project_map_to_streamlines(sft, map_volume, endpoints_only=False):
    """
    Projects a map onto the points of streamlines. The result is a
    data_per_point.

    Parameters
    ----------
    sft: StatefulTractogram
        Input tractogram.
    map_volume: DataVolume
        Input map.
    endpoints_only: bool, optional
        If True, will only project the map_volume onto the endpoints of the
        streamlines (all values along streamlines set to NaN). If False,
        will project the map_volume onto all points of the streamlines.

    Returns
    -------
    streamline_data: List[List]
        The values that could now be associated to a data_per_point key.
        The map_volume projected to each point of the streamlines.
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
        # toDo: there is a double loop. Could be faster?
        for s in sft.streamlines:
            thisstreamline_data = []
            for p in s:
                thisstreamline_data.append(map_volume.get_value_at_coordinate(
                    p[0], p[1], p[2], space=sft.space, origin=sft.origin))

            streamline_data.append(
                np.reshape(thisstreamline_data,
                           (len(thisstreamline_data), dimension)))

    return streamline_data


def project_dpp_to_map(sft, dpp_key, sum_lines=False, endpoints_only=False):
    """
    Saves the values of data_per_point keys to the underlying voxels. Averages
    the values of various streamlines in each voxel. Returns one map per key.
    The streamlines are not preprocessed here. You should probably first
    uncompress your streamlines to have smoother maps.
    Note: If a streamline has two points in the same voxel, it counts twice!

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram
    dpp_key: str
        The data_per_point key to project to a map.
    sum_lines: bool
        Do not average values of streamlines that cross a same voxel; sum them
        instead.
    endpoints_only: bool
        If true, only project the streamline's endpoints.

    Returns
    -------
    the_map: np.ndarray
        The 3D resulting map.
    """
    sft.to_vox()

    # Using to_corner, if we simply floor the coordinates of the point, we find
    # the voxel where it is.
    sft.to_corner()

    # count: could also use compute_tract_counts_map.
    count = np.zeros(sft.dimensions)
    the_map = np.zeros(sft.dimensions)
    for s in range(len(sft)):
        if endpoints_only:
            points = [0, -1]
        else:
            points = range(len(sft.streamlines[s]))

        for p in points:
            x, y, z = sft.streamlines[s][p, :].astype(int)  # Or floor
            count[x, y, z] += 1
            the_map[x, y, z] += sft.data_per_point[dpp_key][s][p]

    if not sum_lines:
        count = np.maximum(count, 1e-6)  # Avoid division by 0
        the_map /= count

    return the_map


def perform_operation_on_dpp(op_name, sft, dpp_name, endpoints_only=False):
    """
    Peforms an operation on the data per point for all streamlines (mean, sum,
    min, max). The operation is applied on each point invidiually, and thus
    makes sense if the data_per_point at each point is a vector. The result is
    a new data_per_point.

    Parameters
    ----------
    op_name: str
        Name of one possible operation (mean, sum, min, max). Will refer to a
        callable that takes a list of streamline data per point (4D) and
        returns a list of streamline data per point.
    sft: StatefulTractogram
        The streamlines used in the operation.
    dpp_name: str
        The name of the data per point to be used in the operation.
        sft.data_per_point[dpp_name][s] must be a 2D vector: (N, M)
        with s, a given streamline; N the number of points; M the number of
        features in the dpp.
    endpoints_only: bool
        If True, will only perform operation on endpoints. Values at other
        points will be set to NaN.

    Returns
    -------
    new_data_per_point: list[np.ndarray]
        The values that could now be associated to a new data_per_point key.
    """
    call_op = OPERATIONS[op_name]
    if endpoints_only:
        new_data_per_point = []
        for s in sft.data_per_point[dpp_name]:
            this_data_per_point = np.nan * np.ones((len(s), 1))
            this_data_per_point[0] = call_op(s[0])
            this_data_per_point[-1] = call_op(s[-1])
            new_data_per_point.append(np.asarray(this_data_per_point)[:, None])
    else:
        new_data_per_point = []
        for s in sft.data_per_point[dpp_name]:
            this_data_per_point = []
            for p in s:
                this_data_per_point.append(call_op(p))
            new_data_per_point.append(np.asarray(this_data_per_point)[:, None])

    return new_data_per_point


def perform_operation_dpp_to_dps(op_name, sft, dpp_name, endpoints_only=False):
    """
    Converts dpp to dps, using a chosen operation.

    Performs an operation across all data_per_points for each streamline (mean,
    sum, min, max). The result is a data_per_streamline.

    If the data_per_point at each point is a vector, operation is done on each
    feature individually. The data_per_streamline will have the same shape.

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
        If True, will only perform operation on endpoints. Other points will be
        ignored in the operation.

    Returns
    -------
    new_data_per_streamline: list
        The values that could now be associated to a new data_per_streamline
        key.
    """
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


def perform_correlation_on_endpoints(sft, dpp_name='metric'):
    """Peforms correlation across endpoints for each streamline. The
    data_per_point at each point must be a vector.

    Parameters
    ----------
    sft: StatefulTractogram
        The streamlines used in the operation.
    dpp_name: str
        The name of the data per point to be used in the operation.

    Returns
    -------
    new_data_per_streamline: List
        The correlation values that could now be associated to a new
        data_per_streamline key.
    """
    new_data_per_streamline = []
    for s in sft.data_per_point[dpp_name]:
        new_data_per_streamline.append(np.corrcoef(s[0], s[-1])[0, 1])

    return new_data_per_streamline


def _stream_mean(array):
    return np.squeeze(np.mean(array, axis=0))


def _stream_sum(array):
    return np.squeeze(np.sum(array, axis=0))


def _stream_min(array):
    return np.squeeze(np.min(array, axis=0))


def _stream_max(array):
    return np.squeeze(np.max(array, axis=0))


OPERATIONS = {
    'mean': _stream_mean,
    'sum': _stream_sum,
    'min': _stream_min,
    'max': _stream_max,
}
