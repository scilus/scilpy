# -*- coding: utf-8 -*-

"""
This module regroups small operations and util functions for tractogram.
Meaning that these operations are applied on the streamlines as wholes (ex,
registration, suffling, etc), not on each point of the streamlines separately /
individually. See scilpy.tractograms.streamline_operations.py for the latter.
"""

from functools import reduce
import itertools
import logging
import random

from dipy.io.stateful_tractogram import set_sft_logger_level, \
    StatefulTractogram, Space
from dipy.io.utils import get_reference_info, is_header_compatible
from dipy.segment.clustering import qbx_and_merge
from dipy.segment.fss import FastStreamlineSearch
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamlinespeed import compress_streamlines
from nibabel.streamlines import TrkFile, TckFile
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

from scilpy.tractanalysis.bundle_operations import uniformize_bundle_sft
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractograms.streamline_operations import smooth_line_gaussian, \
    resample_streamlines_step_size, parallel_transport_streamline, \
    compress_sft, cut_invalid_streamlines, \
    remove_overlapping_points_streamlines, filter_streamlines_by_nb_points
from scilpy.tractograms.streamline_and_mask_operations import \
    cut_streamlines_with_mask
from scilpy.utils.spatial import generate_rotation_matrix

MIN_NB_POINTS = 10
KEY_INDEX = np.concatenate((range(5), range(-1, -6, -1)))


def shuffle_streamlines(sft, rng_seed=None):
    """
    Shuffle the streamlines of a tractogram.

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram to shuffle (will slice the streamline, DPS and DPP).
    rng_seed: int

    Returns
    -------
    shuffled_sft: StatefulTractogram
        The shuffled tractogram.
    """
    indices = np.arange(len(sft.streamlines))
    random.seed(rng_seed)
    random.shuffle(indices)

    streamlines = sft.streamlines[indices]
    data_per_streamline = sft.data_per_streamline[indices]
    data_per_point = sft.data_per_point[indices]

    shuffled_sft = StatefulTractogram.from_sft(
        streamlines, sft,
        data_per_streamline=data_per_streamline,
        data_per_point=data_per_point)
    return shuffled_sft


def shuffle_streamlines_orientation(sft, rng_seed=None):
    """
    Shuffle the orientation of the streamlines. Iterate over streamlines
    and randomly decide (50/50) if the streamline's head and tail should be
    swapped.

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram that will have its streamlines' orientation shuffled.
    rng_seed: int
        Random seed.

    Returns
    -------
    shuffled_sft: StatefulTractogram
        The shuffled tractogram.
    """
    if sft.data_per_point is not None and len(sft.data_per_point) > 0:
        logging.warning('Shuffling streamlines orientation. DPP will be '
                        'lost.')

    rng = np.random.RandomState(rng_seed)
    shuffled_streamlines = []
    for s in sft.streamlines:
        if len(s) < 2:
            shuffled_streamlines.append(s)
        else:
            # flip a coin
            if rng.randint(0, 2):
                shuffled_streamlines.append(s[::-1])
            else:
                shuffled_streamlines.append(s)

    shuffled_sft = StatefulTractogram.from_sft(
        shuffled_streamlines, sft,
        data_per_streamline=sft.data_per_streamline)
    return shuffled_sft


def get_axis_flip_vector(flip_axes):
    """
    Create a flip vector from a list of axes.

    Parameters
    ----------
    flip_axes: list
        List of axes you want to flip

    Returns
    -------
    shift_vector: list[3,]
        Vector with flipped axes
    """
    flip_vector = np.ones(3)
    if 'x' in flip_axes:
        flip_vector[0] = -1.0
    if 'y' in flip_axes:
        flip_vector[1] = -1.0
    if 'z' in flip_axes:
        flip_vector[2] = -1.0

    return flip_vector


def _get_shift_vector(sft):
    dims = sft.space_attributes[1]
    shift_vector = -1.0 * (np.array(dims) / 2.0)

    return shift_vector


def flip_sft(sft, flip_axes):
    """
    Parameters
    ----------
    sft: StatefulTractogram
    flip_axes: List[str]
        The list of axes to flip. Ex: ['x', 'y', 'z']. The axes correspond to
        the coordinates of the sft as it is stored in memory, not to axes
        of the image.

    Returns
    -------
    flipped_sft: StatefulTractogram
    """
    old_space = sft.space
    old_origin = sft.origin
    sft.to_vox()
    sft.to_corner()
    if len(flip_axes) == 0:
        # Could return sft. But creating new SFT (or deep copy).
        flipped_streamlines = sft.streamlines
    else:
        flip_vector = get_axis_flip_vector(flip_axes)
        shift_vector = _get_shift_vector(sft)

        flipped_streamlines = []

        for streamline in sft.streamlines:
            mod_streamline = streamline + shift_vector
            mod_streamline *= flip_vector
            mod_streamline -= shift_vector
            flipped_streamlines.append(mod_streamline)

    new_sft = StatefulTractogram.from_sft(
        flipped_streamlines, sft,
        data_per_point=sft.data_per_point,
        data_per_streamline=sft.data_per_streamline)

    sft.to_space(old_space)
    sft.to_origin(old_origin)
    new_sft.to_space(old_space)
    new_sft.to_origin(old_origin)

    return new_sft


def _get_streamline_key(streamline, precision=None):
    # Use just a few data points as hash key. I could use all the data of
    # the streamlines, but then the complexity grows with the number of
    # points.
    if len(streamline) < MIN_NB_POINTS:
        key = streamline.copy()
    else:
        key = streamline[KEY_INDEX].copy()

    if precision is not None:
        key = np.round(key, precision)

    key.flags.writeable = False

    return key.data.tobytes()


def _hash_streamlines(streamlines, start_index=0, precision=None):
    """
    Produces a dict from streamlines by using the points as keys and the
    indices of the streamlines as values.

    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines used to produce the dict.
    start_index: int, optional
        The index of the first streamline. 0 by default.
    precision: int, optional
        The number of decimals to keep when hashing the points of the
        streamlines. Allows a soft comparison of streamlines. If None, no
        rounding is performed.

    Returns
    -------
    A dict where the keys are streamline points and the values are indices
    starting at start_index.
    """
    keys = [_get_streamline_key(s, precision) for s in streamlines]

    return {k: i for i, k in enumerate(keys, start_index)}


def intersection(left, right):
    """Intersection of two streamlines dict (see hash_streamlines)"""
    return {k: v for k, v in left.items() if k in right}


def difference(left, right):
    """Difference of two streamlines dict (see hash_streamlines)"""
    return {k: v for k, v in left.items() if k not in right}


def union(left, right):
    """Union of two streamlines dict (see hash_streamlines)"""
    return {**left, **right}


def perform_tractogram_operation_on_sft(op_name, sft_list, precision,
                                        no_metadata, fake_metadata):
    """Peforms an operation on a list of tractograms.

    Parameters
    ----------
    op_name: str
        A callable that takes two streamlines dicts as inputs and preduces a
        new streamline dict.
    sft_list: list[StatefulTractogram]
        The tractograms used in the operation.
    precision: int, optional
        The number of decimals to keep when hashing the points of the
        streamlines. Allows a soft comparison of streamlines. If None, no
        rounding is performed. Precision should be in the same space as
        sfts (ex, mm).
    no_metadata: bool
        If true, remove all metadata.
    fake_metadata: bool
        If true, fake metadata for SFTs that do not contain the keys available
        in other SFTs.

    Returns
    -------
    sft: StatefulTractogram
        The final SFT
    """
    streamlines_list = [sft.streamlines for sft in sft_list]
    _, indices = perform_tractogram_operation_on_lines(
        OPERATIONS[op_name], streamlines_list, precision=precision)

    # Current error in dipy prevents concatenation with empty SFT
    # (see PR here to fix: https://github.com/dipy/dipy/pull/2864)
    # Returning empty sft now if that is the case.
    if len(indices) == 0:
        empty_sft = sft_list[0]
        empty_sft.streamlines = []
        return empty_sft, indices

    # Concatenating only the necessary streamlines, with the metadata
    indices_per_sft = []
    streamlines_len_cumsum = [len(sft) for sft in sft_list]
    start = 0
    for nb in streamlines_len_cumsum:
        end = start + nb
        # Switch to int32 for json
        indices_per_sft.append([int(i - start)
                                for i in indices if start <= i < end])
        start = end

    sft_list = [sft[indices_per_sft[i]] for i, sft in enumerate(sft_list)
                if len(indices_per_sft[i]) > 0]

    new_sft = concatenate_sft(sft_list, no_metadata, fake_metadata)
    return new_sft, indices_per_sft


def perform_tractogram_operation_on_lines(operation, streamlines, precision=None,
                                          nb_resample_pts=None, bidirectional=None):
    """Performs a set operation (union, intersection, difference) on a list of lists of streamlines.

    This function acts as a dispatcher for two types of streamline comparisons:

    1. Standard (Hash-based): Recursively applies the operation to the lists of 
       streamlines (first two, then the result with the third, etc.) using a heuristic hash. 
       To save memory and time, this method hashes only the coordinates of the streamline's 
       extremities (first 5 and last 5 points). Valid operations take two streamline 
       dictionaries as input and produce a new streamlines dict.

    2. Robust (FastStreamlineSearch-based): If a "robust" operation is passed 
       (e.g., `union_robust`), it processes all sets simultaneously using DIPY's
       FastStreamlineSearch. This geometric clustering approach allows for soft 
       comparisons, varying point counts, and flipped orientations.

    Parameters
    ----------
    operation: callable
        A callable representing the set operation. Can be a standard hash-based 
        function (e.g., `union`) or a robust geometric function (e.g., `union_robust`).
    streamlines: list of lists of streamlines
        The sets of streamlines used in the operation.
    precision: int, optional
        Determines the strictness of the comparison. For hash-based operations, 
        it is the number of decimals to keep when rounding points. For robust 
        operations, it defines the maximum allowed geometric distance threshold 
        (epsilon = 10^-precision). If None, no rounding is performed for hashes, 
        and robust defaults to 3.
    nb_resample_pts: int, optional
        Used only for 'robust' operations. The number of points to resample 
        the streamlines to before using FastStreamlineSearch.
    bidirectional: bool, optional
        Used only for 'robust' operations. If True, two streamlines are considered 
        similar if they are within the distance threshold, regardless of their 
        anatomical orientation (A-to-B vs B-to-A).

    Returns
    -------
    streamlines: list of `nib.streamline.Streamlines`
        The streamlines obtained after performing the operation on all the
        input streamlines.
    indices: np.ndarray
        The indices of the streamlines that are used in the output.
    """
    if 'robust' in operation.__name__:
        if precision is None:
            precision = 3
        if nb_resample_pts is None:
            nb_resample_pts = 64
        if bidirectional is None:
            bidirectional = True
        return operation(streamlines, precision,
                         nb_resample_pts=nb_resample_pts,
                         bidirectional=bidirectional)
    else:
        # Hash the streamlines using the desired precision.
        indices = np.cumsum([0] + [len(s) for s in streamlines[:-1]])
        hashes = [_hash_streamlines(s, i, precision) for
                  s, i in zip(streamlines, indices)]

        # Perform the operation on the hashes and get the output streamlines.
        to_keep = reduce(operation, hashes)
        all_streamlines = list(itertools.chain(*streamlines))
        indices = np.array(sorted(to_keep.values())).astype(np.uint32)
        streamlines = [all_streamlines[i] for i in indices]
    return streamlines, indices


def intersection_robust(streamlines_list, precision=3,
                        nb_resample_pts=64, bidirectional=True):
    """ Intersection of a list of lists of streamlines """
    if not isinstance(streamlines_list, list):
        streamlines_list = [streamlines_list]

    streamlines_fused, indices = _find_identical_streamlines(
        streamlines_list, epsilon=10**(-precision),
        nb_resample_pts=nb_resample_pts, bidirectional=bidirectional)
    return streamlines_fused[indices], indices


def difference_robust(streamlines_list, precision=3,
                      nb_resample_pts=64, bidirectional=True):
    """ Difference of a list of lists of streamlines from the first element """
    if not isinstance(streamlines_list, list):
        streamlines_list = [streamlines_list]
    streamlines_fused, indices = _find_identical_streamlines(
        streamlines_list, epsilon=10**(-precision), difference_mode=True,
        nb_resample_pts=nb_resample_pts, bidirectional=bidirectional)
    return streamlines_fused[indices], indices


def union_robust(streamlines_list, precision=3,
                 nb_resample_pts=64, bidirectional=True):
    """ Union of a list of lists of streamlines """
    if not isinstance(streamlines_list, list):
        streamlines_list = [streamlines_list]
    streamlines_fused, indices = _find_identical_streamlines(
        streamlines_list, epsilon=10**(-precision), union_mode=True,
        nb_resample_pts=nb_resample_pts, bidirectional=bidirectional,
    )
    return streamlines_fused[indices], indices


def _find_identical_streamlines(streamlines_list, epsilon=0.001,
                                union_mode=False, difference_mode=False,
                                nb_resample_pts=64, bidirectional=True):
    """ Finds identical streamlines across multiple sets and performs a set operation
    (intersection, union, or difference) using DIPY's FastStreamlineSearch.

    This function groups streamlines into connected components based on a maximum 
    allowed geometric distance (epsilon). It then filters these groups based on 
    the requested set operation.

    Note: Regardless of the chosen mode (union, intersection, or difference), 
    this function inherently deduplicates streamlines. It will never return 
    multiple identical copies of a streamline; only a single representative 
    from any matching group is kept.

    Parameters
    -----------
    streamlines_list: list
        List of lists of streamlines or list of ArraySequences.
    epsilon: float, optional
        Maximum allowed distance for matching streamlines (should not go above 1.0).
        Defaults to 0.001.
    union_mode: bool, optional
        If True, performs the union of streamlines (keeps one representative per 
        unique streamline across all sets). Defaults to False.
    difference_mode: bool, optional
        If True, performs the difference of streamlines (keeps unique streamlines 
        that exist ONLY in the first set). Defaults to False.
        Note: If both union_mode and difference_mode are False, the function 
        defaults to intersection mode (keeps streamlines present in ALL sets).
    nb_resample_pts: int, optional
        Number of points to resample the streamlines to for the geometric comparison.
        Defaults to 64.
    bidirectional: bool, optional
        If True, two streamlines are considered similar if one is within epsilon
        of the other, regardless of their orientation. If False, the streamlines
        must be aligned in the same direction. Defaults to True.

    Returns
    --------
    streamlines: ArraySequence
        The concatenated list of all input streamlines.
    indices: np.ndarray
        The indices of the specific streamlines to keep from the concatenated list,
        based on the selected set operation.
    """
    streamlines = ArraySequence(itertools.chain(*streamlines_list))
    nb_streamlines = np.cumsum([len(sft) for sft in streamlines_list])
    nb_streamlines = np.insert(nb_streamlines, 0, 0)

    if union_mode and difference_mode:
        raise ValueError('Cannot use union_mode and difference_mode at the '
                         'same time.')
    intersect_mode = (not union_mode and not difference_mode)

    inversion_val = 1 if union_mode or difference_mode else 0
    streamlines_to_keep = np.ones((len(streamlines),)) * inversion_val

    if difference_mode:
        streamlines_to_keep[nb_streamlines[1]:] = 0

    if len(streamlines) == 0:
        return streamlines, np.array([], dtype=np.uint32)

    fss = FastStreamlineSearch(streamlines,
                               max_radius=epsilon,
                               resampling=nb_resample_pts,
                               bidirectional=bidirectional)

    # adjacency[i, j] = 1 if streamline i is within epsilon of streamline j
    adjacency = fss.radius_search(streamlines, radius=epsilon)

    # Compute connected components:
    # If: A ~ B and B ~ C,
    # then all 3 become one component.
    n_components, labels = connected_components(
        csgraph=adjacency, directed=False)

    # Each iteration handles one group of identical streamlines
    for comp_id in range(n_components):

        group = np.where(labels == comp_id)[0]

        # Determine which sets are present
        sets_present = set(
            np.searchsorted(nb_streamlines, idx, side='right') - 1
            for idx in group)

        if union_mode:
            # Keep first representative
            streamlines_to_keep[group[1:]] = 0

        elif difference_mode:
            # Keep only groups entirely inside first set (only unique streamlines from set 0 survive)
            if not all(idx < nb_streamlines[1] for idx in group):
                streamlines_to_keep[group] = 0
            else:
                # Deduplicate identical streamlines within the first set
                streamlines_to_keep[group[1:]] = 0

        elif intersect_mode:
            # Keep one representative only if present in all sets
            if len(sets_present) == len(streamlines_list):
                rep = min(group)
                streamlines_to_keep[rep] = 1
            else:
                streamlines_to_keep[group] = 0

    return streamlines, np.where(streamlines_to_keep > 0)[0].astype(np.uint32)


def concatenate_sft(sft_list, erase_metadata=False, metadata_fake_init=False):
    """ Concatenate a list of StatefulTractogram together """
    if erase_metadata and metadata_fake_init:
        raise ValueError("You cannot choose both erase_metadata and "
                         "metadata_fake_init")
    if erase_metadata:
        sft_list[0].data_per_point = {}
        sft_list[0].data_per_streamline = {}

    fused_sft = sft_list[0]
    for sft in sft_list[1:]:
        if erase_metadata:
            sft.data_per_point = {}
            sft.data_per_streamline = {}
        elif metadata_fake_init:
            for dps_key in list(sft.data_per_streamline.keys()):
                if dps_key not in sft_list[0].data_per_streamline.keys():
                    del sft.data_per_streamline[dps_key]
            for dpp_key in list(sft.data_per_point.keys()):
                if dpp_key not in sft_list[0].data_per_point.keys():
                    del sft.data_per_point[dpp_key]

            for dps_key in sft_list[0].data_per_streamline.keys():
                if dps_key not in sft.data_per_streamline:
                    arr_shape =\
                        list(sft_list[0].data_per_streamline[dps_key].shape)
                    arr_shape[0] = len(sft)
                    sft.data_per_streamline[dps_key] = np.zeros(arr_shape)
            for dpp_key in sft_list[0].data_per_point.keys():
                if dpp_key not in sft.data_per_point:
                    arr_seq = ArraySequence()
                    arr_seq_shape = list(
                        sft_list[0].data_per_point[dpp_key]._data.shape)
                    arr_seq_shape[0] = len(sft.streamlines._data)
                    arr_seq._data = np.zeros(arr_seq_shape)
                    arr_seq._offsets = sft.streamlines._offsets
                    arr_seq._lengths = sft.streamlines._lengths
                    sft.data_per_point[dpp_key] = arr_seq

        if not metadata_fake_init and \
                not StatefulTractogram.are_compatible(sft, sft_list[0]):
            raise ValueError('Incompatible SFT, check space attributes and '
                             'data_per_point/streamlines.')
        elif not is_header_compatible(sft, sft_list[0]):
            raise ValueError('Incompatible SFT, check space attributes.')

        fused_sft += sft

    return fused_sft


def transform_warp_sft(sft, linear_transfo, target, inverse=False,
                       reverse_op=False, deformation_data=None,
                       remove_invalid=True, cut_invalid=False):
    """ Transform tractogram using an affine Subsequently apply a warp from
    antsRegistration (optional).
    Remove/Cut invalid streamlines to preserve sft validity.

    Parameters
    ----------
    sft: StatefulTractogram
        Stateful tractogram object containing the streamlines to transform.
    linear_transfo: numpy.ndarray
        Linear transformation matrix to apply to the tractogram.
    target: Nifti filepath, image object, header
        Final reference for the tractogram after registration.
    inverse: boolean
        Apply the inverse linear transformation.
    reverse_op: boolean
        Apply both transformation in the reverse order
    deformation_data: np.ndarray
        4D array containing a 3D displacement vector in each voxel.
    remove_invalid: boolean
        Remove the streamlines landing out of the bounding box.
    cut_invalid: boolean
        Cut invalid streamlines rather than removing them. Keep the longest
        segment only.

    Return
    ----------
    new_sft : StatefulTractogram
    """
    # Keep track of the streamlines' original space/origin
    space = sft.space
    origin = sft.origin
    dtype = sft.streamlines._data.dtype

    sft.to_rasmm()
    sft.to_center()

    if len(sft.streamlines) == 0:
        return StatefulTractogram(sft.streamlines, target,
                                  Space.RASMM)

    if inverse:
        linear_transfo = np.linalg.inv(linear_transfo)

    if not reverse_op:
        streamlines = transform_streamlines(sft.streamlines,
                                            linear_transfo)
    else:
        streamlines = sft.streamlines

    if deformation_data is not None:
        if not reverse_op:
            affine, _, _, _ = get_reference_info(target)
        else:
            affine = sft.affine

        # Because of duplication, an iteration over chunks of points is
        # necessary for a big dataset (especially if not compressed)
        streamlines = ArraySequence(streamlines)
        nb_points = len(streamlines._data)
        cur_position = 0
        chunk_size = 1000000
        nb_iteration = int(np.ceil(nb_points/chunk_size))
        inv_affine = np.linalg.inv(affine)

        while nb_iteration > 0:
            max_position = min(cur_position + chunk_size, nb_points)
            points = streamlines._data[cur_position:max_position]

            # To access the deformation information, we need to go in VOX space
            # No need for corner shift since we are doing interpolation
            cur_points_vox = np.array(transform_streamlines(points,
                                                            inv_affine)).T

            x_def = map_coordinates(deformation_data[..., 0],
                                    cur_points_vox.tolist(), order=1)
            y_def = map_coordinates(deformation_data[..., 1],
                                    cur_points_vox.tolist(), order=1)
            z_def = map_coordinates(deformation_data[..., 2],
                                    cur_points_vox.tolist(), order=1)

            # ITK is in LPS and nibabel is in RAS, a flip is necessary for ANTs
            final_points = np.array([-1*x_def, -1*y_def, z_def])
            final_points += np.array(points).T

            streamlines._data[cur_position:max_position] = final_points.T
            cur_position = max_position
            nb_iteration -= 1

    if reverse_op:
        streamlines = transform_streamlines(streamlines, linear_transfo)

    streamlines._data = streamlines._data.astype(dtype)
    new_sft = StatefulTractogram(streamlines, target, space=Space.RASMM,
                                 data_per_point=sft.data_per_point,
                                 data_per_streamline=sft.data_per_streamline)
    if cut_invalid:
        new_sft, _ = cut_invalid_streamlines(new_sft)
    elif remove_invalid:
        new_sft.remove_invalid_streamlines()

    # Move the streamlines back to the original space/origin
    sft.to_space(space)
    sft.to_origin(origin)

    new_sft.to_space(space)
    new_sft.to_origin(origin)

    return new_sft


def compress_streamlines_wrapper(tractogram, error_rate):
    """
    Compresses the streamlines of a tractogram.
    Supports both nibabel.Tractogram dipy.StatefulTractogram
    or list of streamlines.

    Parameters
    ----------
    tractogram: TrkFile, TckFile, ArraySequence, list
        The tractogram to compress.
    error_rate: float
        The maximum distance (in mm) for point displacement during compression.

    Returns
    -------
    compressed_streamlines: list of np.ndarray
        The compressed streamlines.
    """
    if isinstance(tractogram, (TrkFile, TckFile)):
        return lambda: (compress_streamlines(
            s, error_rate) for s in tractogram.streamlines)
    else:
        if hasattr(tractogram, 'streamlines'):
            tractogram = tractogram.streamlines
        return [compress_streamlines(
            s, error_rate) for s in tractogram]


def upsample_tractogram(sft, nb, point_wise_std=None, tube_radius=None,
                        gaussian=None, error_rate=None, seed=None):
    """
    Generates new streamlines by either adding gaussian noise around
    streamlines' points, or by translating copies of existing streamlines
    by a random amount.

    The first streamlines of the returned tractogram are the initial
    streamlines, unchanged (if error_rate is None).

    Parameters
    ----------
    sft : StatefulTractogram
        The tractogram to upsample
    nb : int
        The target number of streamlines in the tractogram.
    point_wise_std : float, optional
        The standard deviation of the gaussian to use to generate point-wise
        noise on the streamlines. If None or zero, this is skipped.
    tube_radius : float, optional
        The radius of the tube used to model the streamlines. If None or zero,
        this is skipped.
    gaussian: float, optional
        The sigma used for smoothing streamlines. Only the newly created
        streamlines are smoothed. If None, streamlines are not smoothed.
    error_rate : float, optional
        The compression error. The whole final tractogram is compressed. If
        None, no compression is done.
    seed: int, optional
        Seed for RNG. If None, uses random seed.

    Returns
    -------
    new_sft : StatefulTractogram
        The upsampled tractogram.
    """
    rng = np.random.default_rng(seed)

    if nb < len(sft):
        logging.warning("Wrong call of this upsampling method: the "
                        "tractogram already contains more streamlines than "
                        "wanted.")
    if nb <= len(sft):
        return sft

    nb = nb - len(sft)

    # Get the streamlines that will serve as a base for new ones
    indices = rng.choice(len(sft), nb, replace=True)
    unique_indices, count = np.unique(indices, return_counts=True)
    resampled_sft = resample_streamlines_step_size(sft[unique_indices], 1)

    # For all selected streamlines, add noise and smooth
    new_streamlines = sft.streamlines
    for s, c in zip(resampled_sft.streamlines, count):
        # 1. Translate the streamline, up to a tube_radius distance.
        if tube_radius is not None and tube_radius > 0:
            new_s = parallel_transport_streamline(s, c, tube_radius)
        else:
            new_s = [s] * c

        # 2. Add point-wise noise.
        if point_wise_std is not None and point_wise_std > 0:
            # Generate smooth noise_factor
            noise = rng.normal(loc=0, scale=point_wise_std, size=len(s))

            # Instead of generating random noise, we fit a polynomial to
            # the noise and use it to generate a spatially smooth noise
            # along the streamline (simply to avoid sharp changes in the
            # noise factor).
            x = np.arange(len(noise))
            poly_coeffs = np.polyfit(x, noise, 3)
            polynomial = Polynomial(poly_coeffs[::-1])
            noise_factor = polynomial(x)

            vec = s - new_s
            norm = np.linalg.norm(vec, axis=0)
            if np.any(norm == 0):
                vec = np.ones_like(vec)
                norm = np.linalg.norm(vec, axis=0)
            vec /= norm

            new_s += vec * np.expand_dims(noise_factor, axis=1)

            # Result is of shape [c, len(s), 3]. Splitting back
            new_s = list(new_s)

        # 3. Smooth the result.
        if gaussian:
            new_s = [smooth_line_gaussian(s, gaussian) for s in new_s]

        new_streamlines.extend(new_s)

    if error_rate:
        compressed_streamlines = compress_streamlines_wrapper(new_streamlines,
                                                              error_rate)
    else:
        compressed_streamlines = new_streamlines

    new_sft = StatefulTractogram.from_sft(compressed_streamlines, sft)
    return new_sft


def split_sft_sequentially(orig_sft, chunk_sizes):
    """
    Divides a stateful tractogram into n sub-tractograms of sizes defined by
    chunk_sizes. Streamlines are separated sequentially from the initial
    streamlines.

    Parameters
    ----------
    orig_sft: StatefulTractogram
        Initial tractogram to subdivide.
    chunk_sizes: list[int]
        Number of streamlines to keep per chunk.

    Return
    ------
    all_chunks: list[StatefulTractogram]
        The list of sub-tractograms as sfts. The number of tractograms returned
        is len(chunk_sizes).
    """
    if sum(chunk_sizes) > len(orig_sft):
        raise ValueError("You asked for more streamlines than are available.")

    nb_chunks = len(chunk_sizes)

    curr = 0
    sfts = []
    for i in range(nb_chunks):
        nb_str = chunk_sizes[i]
        sfts.append(orig_sft[curr:curr + nb_str])
        curr += chunk_sizes[i]

    return sfts


def split_sft_randomly(orig_sft, chunk_sizes, rng_seed,
                       return_indices_only=False):
    """
    Divides a stateful tractogram into n sub-tractograms of sizes defined by
    chunk_sizes. Streamlines are separated randomly from the initial
    streamlines.

    Parameters
    ----------
    orig_sft: StatefulTractogram
        Initial tractogram to subdivide
    chunk_sizes: int or list[int]
        Number of streamlines to keep (per sub-tractogram if it is a list).
    rng_seed: int
        Random seed.
    return_indices_only: bool
        If true, return a random list of indices. Else, return the Stateful
        Tractogram containing the chosen streamlines.

    Return
    ------
    all_chunks: list[StatefulTractogram] or list[list[int]]
        The list of sub-tractograms as sfts. The number of tractograms returned
        is len(chunk_sizes) + 1, where the last item of the list contains
        streamlines that were not included in any.
        (Or the lists of indices if return_indices_only.)
    """
    if isinstance(chunk_sizes, int):
        chunk_sizes = [chunk_sizes]

    if sum(chunk_sizes) > len(orig_sft):
        raise ValueError("You asked for more streamlines than are available.")

    # Shuffle all streamline indices
    rng = np.random.RandomState(rng_seed)
    ind = np.arange(len(orig_sft.streamlines))
    rng.shuffle(ind)

    # Separate indices.
    final_indices = []
    start = 0
    for next_nb_streamlines in chunk_sizes:
        sub_ind = ind[start:start+next_nb_streamlines]
        final_indices.append(sub_ind)
        start += next_nb_streamlines

    # Append indices not included in any chunk
    final_indices.append(ind[start:])

    if return_indices_only:
        return final_indices

    # Format as sft
    all_sfts = []
    for i in range(len(chunk_sizes) + 1):
        all_sfts.append(orig_sft[final_indices[i]])

    return all_sfts


def split_sft_randomly_per_cluster(orig_sft, chunk_sizes, seed, thresholds):
    """
    Divides a stateful tractogram into n sub-tractograms of sizes defined by
    chunk_sizes. Streamlines are separated randomly from each Quickbundle
    cluster created from the initial streamlines (trying to help
    the randomization to ensure there are streamlines from all bundles in each
    subset).

    Parameters
    ----------
    orig_sft: StatefulTractogram
        Initial tractogram to subdivide
    chunk_sizes: list[int]
        Number of streamlines to keep per chunk. We will ensure that the number
        of streamlines kept per cluster is proportional to the cluster's size.
        Final number will be a good approximation of nb_streamlines, but not
        exact.
    seed: int
        Random seed.
    thresholds: list[float]
        QBx threshold values. Suggestion: [40, 30, 20].

    Returns
    -------
    all_sfts: list[StatefulTractogram]
        The list of sub-tractograms as sfts. The number of tractograms returned
        is len(chunk_sizes) + 1, where the last item of the list contains
        streamlines that were not included in any.
    """

    if sum(chunk_sizes) > len(orig_sft):
        raise ValueError("You asked for more streamlines than are available.")

    # Percent of streamlines to keep per chunk.
    nb_chunks = len(chunk_sizes)
    percent_kept_per_chunk = [nb / len(orig_sft) for nb in chunk_sizes]

    logging.debug("Computing QBx")
    rng = np.random.RandomState(seed)
    clusters = qbx_and_merge(orig_sft.streamlines, thresholds, nb_pts=20,
                             verbose=False, rng=rng)

    logging.info("Done. Now getting list of indices in each of the {} "
                 "cluster.".format(len(clusters)))
    total_indices = [[] for _ in range(nb_chunks + 1)]
    for cluster in clusters:
        if len(cluster.indices) > 1:
            cluster_sft = orig_sft[cluster.indices]
            size_cluster = len(cluster.indices)
            chunk_sizes_in_cluster = \
                [round(p * size_cluster) for p in percent_kept_per_chunk]

            # If rounding created too many streamlines, removing some from the
            # last chunk.
            while sum(chunk_sizes_in_cluster) > size_cluster:
                chunk_sizes_in_cluster[-1] -= 1

            all_chunks_inds_in_cluster = split_sft_randomly(
                cluster_sft, chunk_sizes_in_cluster, seed,
                return_indices_only=True)

            assert len(all_chunks_inds_in_cluster) == nb_chunks + 1

            for i in range(nb_chunks + 1):
                chunk_orig_inds = [cluster.indices[ind] for ind in
                                   all_chunks_inds_in_cluster[i]]
                total_indices[i].extend(chunk_orig_inds)

    final_sfts = [orig_sft[inds] for inds in total_indices]

    return final_sfts


def subsample_streamlines_alter(sft, min_dice=0.90, epsilon=0.01,
                                baseline_sft=None):
    """
    Function to subsample streamlines based on a dice similarity metric.
    The function will keep removing streamlines until the dice similarity
    between the original and the subsampled tractogram is close to min_dice.

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram to subsample.
    min_dice: float
        The minimum dice similarity to reach before stopping the subsampling.
    epsilon: float
        Stopping criteria for convergence. The maximum difference between the
        dice similarity and min_dice.
    baseline_sft: StatefulTractogram
        The tractogram to use as a reference for the dice similarity. If None,
        the original tractogram will be used.

    Returns
    -------
    new_sft: StatefulTractogram
        The tractogram with a subset of streamlines in the same space as the
        input tractogram.
    """
    # Import in function to avoid circular import error
    from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel
    set_sft_logger_level(logging.ERROR)
    space = sft.space
    origin = sft.origin

    sft.to_vox()
    sft.to_corner()
    if baseline_sft is None:
        original_density_map = compute_tract_counts_map(sft.streamlines,
                                                        sft.dimensions)
    else:
        baseline_sft.to_vox()
        baseline_sft.to_corner()
        original_density_map = compute_tract_counts_map(baseline_sft.streamlines,
                                                        sft.dimensions)
    dice = 1.0
    init_pick_min = 0
    init_pick_max = len(sft)
    previous_to_pick = None
    while dice > min_dice or np.abs(dice - min_dice) > epsilon:
        to_pick = init_pick_min + (init_pick_max - init_pick_min) // 2
        if to_pick == previous_to_pick:
            logging.warning('No more streamlines to pick, not converging.')
            break
        previous_to_pick = to_pick

        indices = np.random.choice(len(sft), to_pick, replace=False)
        streamlines = sft.streamlines[indices]
        curr_density_map = compute_tract_counts_map(streamlines,
                                                    sft.dimensions)
        dice, _ = compute_dice_voxel(original_density_map, curr_density_map)
        logging.debug(f'Subsampled {to_pick} streamlines, dice: {dice}')

        if dice < min_dice:
            init_pick_min = to_pick
        else:
            init_pick_max = to_pick

    new_sft = StatefulTractogram.from_sft(streamlines, sft)
    new_sft.to_space(space)
    new_sft.to_origin(origin)
    return new_sft


def cut_streamlines_alter(sft, min_dice=0.90, epsilon=0.01, from_end=False):
    """
    Cut streamlines based on a dice similarity metric.
    The function will keep removing points from the streamlines until the dice
    similarity between the original and the cut tractogram is close
    to min_dice.

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram to cut.
    min_dice: float
        The minimum dice similarity to reach before stopping the cutting.
    epsilon: float
        Stopping criteria for convergence. The maximum difference between the
        dice similarity and min_dice.
    from_end: bool
        If True, the streamlines will be cut from the end. Else,
        the streamlines will be cut from the start.

    Returns
    -------
    new_sft: StatefulTractogram
        The tractogram with cut streamlines in the same space as the input
        tractogram.
    """
    # Import in function to avoid circular import error
    from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel
    set_sft_logger_level(logging.ERROR)
    space = sft.space
    origin = sft.origin

    # Uniformize endpoints to cut consistently from one end only
    uniformize_bundle_sft(sft, swap=from_end)
    sft = resample_streamlines_step_size(sft, 0.5)
    sft.to_vox()
    sft.to_corner()
    original_density_map = compute_tract_counts_map(sft.streamlines,
                                                    sft.dimensions)

    # Initialize the dice value and the cut percentage for dichotomic search
    dice = 1.0
    init_cut_min = 0
    init_cut_max = 1.0
    previous_to_pick = None
    while dice > min_dice or np.abs(dice - min_dice) > epsilon:
        to_pick = init_cut_min + (init_cut_max - init_cut_min) / 2
        if to_pick == previous_to_pick:
            logging.warning('No more points to pick, not converging.')
            break
        previous_to_pick = to_pick

        streamlines = []
        for streamline in sft.streamlines:
            pos_to_pick = int(len(streamline) * to_pick)
            streamline = streamline[:pos_to_pick]
            streamlines.append(streamline)
        curr_density_map = compute_tract_counts_map(streamlines,
                                                    sft.dimensions)
        dice, _ = compute_dice_voxel(original_density_map, curr_density_map)
        logging.debug(f'Cut {to_pick * 100}% of the streamlines, dice: {dice}')

        if dice < min_dice:
            init_cut_min = to_pick
        else:
            init_cut_max = to_pick

    new_sft = StatefulTractogram.from_sft(streamlines, sft)
    new_sft.to_space(space)
    new_sft.to_origin(origin)
    return compress_sft(new_sft)


def replace_streamlines_alter(sft, min_dice=0.90, epsilon=0.01):
    """
    Replace streamlines based on a dice similarity metric.
    The function will upsample the streamlines (with parallel transport),
    then downsample them until the dice similarity is close to min_dice.
    This effectively replaces the streamlines with new ones.

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram to replace streamlines from.
    min_dice: float
        The minimum dice similarity to reach before stopping the replacement.
    epsilon: float
        Stopping criteria for convergence. The maximum difference between the
        dice similarity and min_dice.

    Returns
    -------
    new_sft: StatefulTractogram
        The tractogram with replaced streamlines in the same space as the input
        tractogram.
    """
    set_sft_logger_level(logging.ERROR)

    logging.debug('Upsampling the streamlines by a factor 2x to then '
                  'downsample.')
    upsampled_sft = upsample_tractogram(sft, len(sft) * 2, point_wise_std=0.5,
                                        tube_radius=1.0, gaussian=None,
                                        error_rate=0.1, seed=1234)
    return subsample_streamlines_alter(upsampled_sft, min_dice, epsilon,
                                       baseline_sft=sft)


def trim_streamlines_alter(sft, min_dice=0.90, epsilon=0.01):
    """
    Trim streamlines based on a dice similarity metric.
    The function will remove low density voxels to trim streamlines until the
    similarity between the original and the trimmed tractogram is close to
    min_dice.

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram to trim.
    min_dice: float
        The minimum dice similarity to reach before stopping the trimming.
    epsilon: float
        Stopping criteria for convergence. The maximum difference between the
        dice similarity and min_dice.

    Returns
    -------
    new_sft: StatefulTractogram
        The tractogram with trimmed streamlines in the same space as the input
        tract
    """
    # Import in function to avoid circular import error
    from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel
    set_sft_logger_level(logging.ERROR)
    space = sft.space
    origin = sft.origin

    sft.to_vox()
    sft.to_corner()
    original_density_map = compute_tract_counts_map(
        sft.streamlines, sft.dimensions).astype(np.uint64)
    thr_density = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    thr_pos = 0
    voxels_to_remove = np.where(
        (original_density_map <= thr_density[thr_pos]) &
        (original_density_map > 0))

    # Initialize the dice value and the number of voxels to pick
    dice = 1.0
    previous_dice = 0.0
    init_trim_min = 0
    init_trim_max = np.count_nonzero(voxels_to_remove[0])
    previous_to_pick = None

    while dice > min_dice or np.abs(dice - previous_dice) > epsilon:
        to_pick = init_trim_min + (init_trim_max - init_trim_min) // 2
        if to_pick == previous_to_pick or \
                np.abs(dice - previous_dice) < epsilon:
            # If too few voxels are picked, increase the threshold
            # and reinitialize the picking

            if np.abs(dice - min_dice) > epsilon and \
                    thr_pos < len(thr_density) - 1:
                thr_pos += 1
                logging.debug(f'Increasing threshold density to '
                              f'{thr_density[thr_pos]}.')

                voxels_to_remove = np.where(
                    (original_density_map <= thr_density[thr_pos]) &
                    (original_density_map > 0))
                init_trim_min = 0
                init_trim_max = np.count_nonzero(voxels_to_remove[0])
                dice = 1.0
                previous_dice = 0.0
                previous_to_pick = None
                continue
            else:
                break
        previous_to_pick = to_pick

        voxel_to_remove = np.where(
            (original_density_map <= thr_density[thr_pos]) &
            (original_density_map > 0))
        indices = np.random.choice(np.count_nonzero(voxel_to_remove[0]),
                                   to_pick, replace=False)
        voxel_to_remove = tuple(np.array(voxel_to_remove).T[indices].T)
        mask = original_density_map.copy()
        mask[voxel_to_remove] = 0

        # set logger level to ERROR to avoid logging from cut_outside_of_mask
        log_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        new_sft = cut_streamlines_with_mask(sft, mask, min_len=10)
        # reset logger level
        logging.getLogger().setLevel(log_level)

        curr_density_map = compute_tract_counts_map(new_sft.streamlines,
                                                    sft.dimensions)
        previous_dice = dice
        dice, _ = compute_dice_voxel(original_density_map, curr_density_map)
        logging.debug(f'Trimmed {to_pick} voxels at density '
                      f'{thr_density[thr_pos]}, dice: {dice}')

        if dice < min_dice:
            init_trim_max = to_pick
        else:
            init_trim_min = to_pick

    new_sft.to_space(space)
    new_sft.to_origin(origin)
    return new_sft


def transform_streamlines_alter(sft, min_dice=0.90, epsilon=0.01):
    """
    The function will apply random rotations to the streamlines until the dice
    similarity between the original and the transformed tractogram is close to
    min_dice.

    Start with a large XYZ rotation, then reduce the rotation step by half one
    axis at a time until the dice similarity is close to min_dice.

    Parameters
    ----------
    sft: StatefulTractogram
        The tractogram to transform.
    min_dice: float
        The minimum dice similarity to reach before stopping the transformation.
    epsilon: float
        Stopping criteria for convergence. The maximum difference between the
        dice similarity and min_dice.

    Returns
    -------
    new_sft: StatefulTractogram
        The tractogram with transformed streamlines in the same space as the
        input tractogram.
    """
    # Import in function to avoid circular import error
    from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel
    set_sft_logger_level(logging.ERROR)
    space = sft.space
    origin = sft.origin

    sft.to_vox()
    sft.to_corner()
    original_density_map = compute_tract_counts_map(sft.streamlines,
                                                    sft.dimensions)

    # Initialize the dice value and angles to pick
    dice = 1.0
    angle_min = [0.0, 0.0, 0.0]
    angle_max = [0.1, 0.1, 0.1]
    previous_dice = None
    last_pick = np.array([0.0, 0.0, 0.0])
    rand_val = np.random.rand(3) * angle_max[0]
    axis_choices = np.random.choice(3, 3, replace=False)
    axis = 0
    while dice > min_dice or np.abs(dice - min_dice) > epsilon:
        init_angle_min = angle_min[axis]
        init_angle_max = angle_max[axis]
        to_pick = init_angle_min + (init_angle_max - init_angle_min) / 2

        # Generate a 4x4 matrix from random euler angles
        rand_val = np.array(angle_max)
        rand_val[axis] = to_pick

        angles = rand_val * 2 * np.pi
        rot_mat = generate_rotation_matrix(angles)
        streamlines = transform_streamlines(sft.streamlines, rot_mat)

        # Remove invalid streamlines to avoid numerical issues
        curr_sft = StatefulTractogram.from_sft(streamlines, sft)
        curr_sft, _ = cut_invalid_streamlines(curr_sft)
        curr_sft = filter_streamlines_by_nb_points(curr_sft, min_nb_points=2)
        curr_sft = remove_overlapping_points_streamlines(curr_sft)

        curr_density_map = compute_tract_counts_map(curr_sft.streamlines,
                                                    sft.dimensions)
        dice, _ = compute_dice_voxel(original_density_map, curr_density_map)
        logging.debug(f'Transformed {np.round(to_pick * 360, 6)} degree '
                      f'on axis {axis}, dice: {dice}')
        last_pick[axis] = to_pick

        if dice < min_dice:
            angle_max[axis] = to_pick
        else:
            angle_min[axis] = to_pick

        if (previous_dice is not None) \
                and np.abs(dice - previous_dice) < epsilon / 2:
            logging.debug('Not converging, switching axis.\n')
            axis_choices = np.roll(axis_choices, 1)
            axis = axis_choices[0]
        previous_dice = dice

    logging.debug(f'\nFinal angles: {last_pick * 360} at dice: {dice}')
    curr_sft.to_space(space)
    curr_sft.to_origin(origin)
    return curr_sft, rot_mat


OPERATIONS = {
    'difference_robust': difference_robust,
    'intersection_robust': intersection_robust,
    'union_robust': union_robust,
    'difference': difference,
    'intersection': intersection,
    'union': union,
    'concatenate': 'concatenate',
    'lazy_concatenate': 'lazy_concatenate'
}
