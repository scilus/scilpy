# -*- coding: utf-8 -*-
import copy
import logging

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamlinespeed import compress_streamlines
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from scilpy.io.utils import load_matrix_in_any_format
from scilpy.tractanalysis.features import get_streamlines_centroid
from scilpy.tractograms.streamline_and_mask_operations import \
    get_endpoints_density_map


def uniformize_bundle_sft(sft, axis=None, ref_bundle=None, swap=False):
    """Uniformize the streamlines in the given tractogram.

    Parameters
    ----------
    sft: StatefulTractogram
         The tractogram that contains the list of streamlines to be uniformized
    axis: int, optional
        Orient endpoints in the given axis
    ref_bundle: streamlines
        Orient endpoints the same way as this bundle (or centroid)
    swap: boolean, optional
        Swap the orientation of streamlines
    """
    old_space = sft.space
    old_origin = sft.origin
    sft.to_vox()
    sft.to_corner()
    density = get_endpoints_density_map(sft, point_to_select=3)
    indices = np.argwhere(density > 0)
    kmeans = KMeans(n_clusters=2, random_state=0, copy_x=True,
                    n_init=20).fit(indices)

    labels = np.zeros(density.shape)
    for i in range(len(kmeans.labels_)):
        labels[tuple(indices[i])] = kmeans.labels_[i]+1

    k_means_centers = kmeans.cluster_centers_
    main_dir_barycenter = np.argmax(
        np.abs(k_means_centers[0] - k_means_centers[-1]))

    if len(sft.streamlines) > 0:
        axis_name = ['x', 'y', 'z']
        if axis is None or ref_bundle is not None:
            if ref_bundle is not None:
                ref_bundle.to_vox()
                ref_bundle.to_corner()
                centroid = get_streamlines_centroid(ref_bundle.streamlines,
                                                    20)[0]
            else:
                centroid = get_streamlines_centroid(sft.streamlines, 20)[0]
            main_dir_ends = np.argmax(np.abs(centroid[0] - centroid[-1]))
            main_dir_displacement = np.argmax(
                np.abs(np.sum(np.gradient(centroid, axis=0), axis=0)))

            if main_dir_displacement != main_dir_ends \
                    or main_dir_displacement != main_dir_barycenter:
                logging.info('Ambiguity in orientation, you should use --axis')
            axis = axis_name[main_dir_displacement]
        logging.info('Orienting endpoints in the {} axis'.format(axis))
        axis_pos = axis_name.index(axis)

        if bool(k_means_centers[0][axis_pos] >
                k_means_centers[1][axis_pos]) ^ bool(swap):
            labels[labels == 1] = 3
            labels[labels == 2] = 1
            labels[labels == 3] = 2

        for i in range(len(sft.streamlines)):
            if ref_bundle:
                res_centroid = set_number_of_points(centroid, 20)
                res_streamlines = set_number_of_points(sft.streamlines[i], 20)
                norm_direct = np.sum(
                    np.linalg.norm(res_centroid - res_streamlines, axis=0))
                norm_flip = np.sum(
                    np.linalg.norm(res_centroid - res_streamlines[::-1], axis=0))
                if bool(norm_direct > norm_flip) ^ bool(swap):
                    sft.streamlines[i] = sft.streamlines[i][::-1]
                    for key in sft.data_per_point[i]:
                        sft.data_per_point[key][i] = \
                            sft.data_per_point[key][i][::-1]
            else:
                # Bitwise XOR
                if bool(labels[tuple(sft.streamlines[i][0].astype(int))] >
                        labels[tuple(sft.streamlines[i][-1].astype(int))]) ^ bool(swap):
                    sft.streamlines[i] = sft.streamlines[i][::-1]
                    for key in sft.data_per_point[i]:
                        sft.data_per_point[key][i] = \
                            sft.data_per_point[key][i][::-1]
    sft.to_space(old_space)
    sft.to_origin(old_origin)


def uniformize_bundle_sft_using_mask(sft, mask, swap=False):
    """Uniformize the streamlines in the given tractogram so head is closer to
    to a region of interest.

    Parameters
    ----------
    sft: StatefulTractogram
         The tractogram that contains the list of streamlines to be uniformized
    mask: np.ndarray
        Mask to use as a reference for the ROI.
    swap: boolean, optional
        Swap the orientation of streamlines
    """

    # barycenter = np.average(np.argwhere(mask), axis=0)
    old_space = sft.space
    old_origin = sft.origin
    sft.to_vox()
    sft.to_corner()

    tree = cKDTree(np.argwhere(mask))
    for i in range(len(sft.streamlines)):
        head_dist = tree.query(sft.streamlines[i][0])[0]
        tail_dist = tree.query(sft.streamlines[i][-1])[0]
        if bool(head_dist > tail_dist) ^ bool(swap):
            sft.streamlines[i] = sft.streamlines[i][::-1]
            for key in sft.data_per_point[i]:
                sft.data_per_point[key][i] = \
                    sft.data_per_point[key][i][::-1]

    sft.to_space(old_space)
    sft.to_origin(old_origin)


def clip_and_normalize_data_for_cmap(args, data):
    if args.LUT:
        LUT = load_matrix_in_any_format(args.LUT)
        for i, val in enumerate(LUT):
            data[data == i+1] = val

    if args.min_range is not None or args.max_range is not None:
        data = np.clip(data, args.min_range, args.max_range)

    # get data values range
    if args.min_cmap is not None:
        lbound = args.min_cmap
    else:
        lbound = np.min(data)
    if args.max_cmap is not None:
        ubound = args.max_cmap
    else:
        ubound = np.max(data)

    if args.log:
        data[data > 0] = np.log10(data[data > 0])

    # normalize data between 0 and 1
    data -= lbound
    data = data / ubound if ubound > 0 else data
    return data, lbound, ubound


def get_color_streamlines_from_angle(sft, args):
    """Color streamlines according to their length.

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram that contains the list of streamlines to be colored
    args: NameSpace
        The colormap options.

    Returns
    -------
    color: np.ndarray
        An array of shape (nb_streamlines, 3) containing the RGB values of
        streamlines
    lbound: float
        Minimal value
    ubound: float
        Maximal value
    """
    angles = []
    for i in range(len(sft.streamlines)):
        dirs = np.diff(sft.streamlines[i], axis=0)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        cos_angles = np.sum(dirs[:-1, :] * dirs[1:, :], axis=1)
        # Resolve numerical instability
        cos_angles = np.minimum(np.maximum(-1.0, cos_angles), 1.0)
        line_angles = [0.0] + list(np.arccos(cos_angles)) + [0.0]
        angles.extend(line_angles)

    angles = np.rad2deg(angles)

    return clip_and_normalize_data_for_cmap(args, angles)


def get_color_streamlines_along_length(sft, args):
    """Color streamlines according to their length.

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram that contains the list of streamlines to be colored
    args: NameSpace
        The colormap options.

    Returns
    -------
    color: np.ndarray
        An array of shape (nb_streamlines, 3) containing the RGB values of
        streamlines
    lbound: int
        Minimal value
    ubound: int
        Maximal value
    """
    positions = []
    for i in range(len(sft.streamlines)):
        positions.extend(list(np.linspace(0, 1, len(sft.streamlines[i]))))

    return clip_and_normalize_data_for_cmap(args, positions)


def filter_tractogram_data(tractogram, streamline_ids):
    """ Filter tractogram according to streamline ids and keep the data

    Parameters:
    -----------
    tractogram: StatefulTractogram
        Tractogram containing the data to be filtered
    streamline_ids: array_like
        List of streamline ids the data corresponds to

    Returns:
    --------
    new_tractogram: Tractogram or StatefulTractogram
        Returns a new tractogram with only the selected streamlines
        and data
    """

    streamline_ids = np.asarray(streamline_ids, dtype=int)

    assert np.all(
        np.in1d(streamline_ids, np.arange(len(tractogram.streamlines)))
    ), "Received ids outside of streamline range"

    new_streamlines = tractogram.streamlines[streamline_ids]
    new_data_per_streamline = tractogram.data_per_streamline[streamline_ids]
    new_data_per_point = tractogram.data_per_point[streamline_ids]

    # Could have been nice to deepcopy the tractogram modify the attributes in
    # place instead of creating a new one, but tractograms cant be subsampled
    # if they have data

    return StatefulTractogram.from_sft(
        new_streamlines,
        tractogram,
        data_per_point=new_data_per_point,
        data_per_streamline=new_data_per_streamline)


def compress_sft(sft, tol_error=0.01):
    """ Compress a stateful tractogram. Uses Dipy's compress_streamlines, but
    deals with space better.

    Dipy's description:
    The compression consists in merging consecutive segments that are
    nearly collinear. The merging is achieved by removing the point the two
    segments have in common.

    The linearization process [Presseau15]_ ensures that every point being
    removed are within a certain margin (in mm) of the resulting streamline.
    Recommendations for setting this margin can be found in [Presseau15]_
    (in which they called it tolerance error).

    The compression also ensures that two consecutive points won't be too far
    from each other (precisely less or equal than `max_segment_length`mm).
    This is a tradeoff to speed up the linearization process [Rheault15]_. A
    low value will result in a faster linearization but low compression,
    whereas a high value will result in a slower linearization but high
    compression.

    [Presseau C. et al., A new compression format for fiber tracking datasets,
    NeuroImage, no 109, 73-83, 2015.]

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to compress.
    tol_error: float (optional)
        Tolerance error in mm (default: 0.01). A rule of thumb is to set it
        to 0.01mm for deterministic streamlines and 0.1mm for probabilitic
        streamlines.

    Returns
    -------
    compressed_sft : StatefulTractogram
    """
    # Go to world space
    orig_space = sft.space
    sft.to_rasmm()

    # Compress streamlines
    compressed_streamlines = compress_streamlines(sft.streamlines,
                                                  tol_error=tol_error)
    if sft.data_per_point is not None and sft.data_per_point.keys():
        logging.warning("Initial StatefulTractogram contained data_per_point. "
                        "This information will not be carried in the final "
                        "tractogram.")

    compressed_sft = StatefulTractogram.from_sft(
        compressed_streamlines, sft,
        data_per_streamline=sft.data_per_streamline)

    # Return to original space
    compressed_sft.to_space(orig_space)

    return compressed_sft


def cut_invalid_streamlines(sft):
    """ Cut streamlines so their longest segment are within the bounding box.
    This function keeps the data_per_point and data_per_streamline.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to remove invalid points from.

    Returns
    -------
    new_sft : StatefulTractogram
        New object with the invalid points removed from each streamline.
    cutting_counter : int
        Number of streamlines that were cut.
    """
    if not len(sft):
        return sft, 0

    # Keep track of the streamlines' original space/origin
    space = sft.space
    origin = sft.origin

    sft.to_vox()
    sft.to_corner()

    copy_sft = copy.deepcopy(sft)
    epsilon = 0.001
    indices_to_remove, _ = copy_sft.remove_invalid_streamlines()

    new_streamlines = []
    new_data_per_point = {}
    new_data_per_streamline = {}
    for key in sft.data_per_point.keys():
        new_data_per_point[key] = []
    for key in sft.data_per_streamline.keys():
        new_data_per_streamline[key] = []

    cutting_counter = 0
    for ind in range(len(sft.streamlines)):
        # No reason to try to cut if all points are within the volume
        if ind in indices_to_remove:
            best_pos = [0, 0]
            cur_pos = [0, 0]
            for pos, point in enumerate(sft.streamlines[ind]):
                if (point < epsilon).any() or \
                        (point >= sft.dimensions - epsilon).any():
                    cur_pos = [pos+1, pos+1]
                if cur_pos[1] - cur_pos[0] > best_pos[1] - best_pos[0]:
                    best_pos = cur_pos
                cur_pos[1] += 1

            if not best_pos == [0, 0]:
                new_streamlines.append(
                    sft.streamlines[ind][best_pos[0]:best_pos[1]-1])
                cutting_counter += 1
                for key in sft.data_per_streamline.keys():
                    new_data_per_streamline[key].append(
                        sft.data_per_streamline[key][ind])
                for key in sft.data_per_point.keys():
                    new_data_per_point[key].append(
                        sft.data_per_point[key][ind][best_pos[0]:best_pos[1]-1])
            else:
                logging.warning('Streamlines entirely out of the volume.')
        else:
            new_streamlines.append(sft.streamlines[ind])
            for key in sft.data_per_streamline.keys():
                new_data_per_streamline[key].append(
                    sft.data_per_streamline[key][ind])
            for key in sft.data_per_point.keys():
                new_data_per_point[key].append(sft.data_per_point[key][ind])
    new_sft = StatefulTractogram.from_sft(new_streamlines, sft,
                                          data_per_streamline=new_data_per_streamline,
                                          data_per_point=new_data_per_point)

    # Move the streamlines back to the original space/origin
    sft.to_space(space)
    sft.to_origin(origin)

    new_sft.to_space(space)
    new_sft.to_origin(origin)

    return new_sft, cutting_counter
