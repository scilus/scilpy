#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import reduce
import itertools

import numpy as np


MIN_NB_POINTS = 10
KEY_INDEX = np.concatenate((range(5), range(-1, -6, -1)))


def get_streamline_key(streamline, precision=None):
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


def hash_streamlines(streamlines, start_index=0, precision=None):
    """Produces a dict from streamlines

    Produces a dict from streamlines by using the points as keys and the
    indices of the streamlines as values.

    Args:
        streamlines (list of ndarray) : The list of streamlines used to
            produce the dict.
        start_index (int, optional) : The index of the first streamline.
            0 by default.
        precision (int, optional) : The number of decimals to keep when
            hashing the points of the streamlines. Allows a soft
            comparison of streamlines. If None, no rounding is performed.

    Returns:
        A dict where the keys are streamline points and the values are
        indices starting at start_index.

    """

    keys = [get_streamline_key(s, precision) for s in streamlines]
    return {k: i for i, k in enumerate(keys, start_index)}


def intersection(left, right):
    """Intersection of two streamlines dict (see hash_streamlines)"""
    return {k: v for k, v in left.items() if k in right}


def subtraction(left, right):
    """Subtraction of two streamlines dict (see hash_streamlines)"""
    return {k: v for k, v in left.items() if k not in right}


def perform_streamlines_operation(operation, streamlines, precision=None):
    """Peforms an operation on a list of list of streamlines

    Given a list of list of streamlines, this function applies the operation
    to the first two lists of streamlines. The result in then used recursively
    with the third, fourth, etc. lists of streamlines.

    A valid operation is any function that takes two streamlines dict as input
    and produces a new streamlines dict (see hash_streamlines). Union,
    subtraction, and intersection are valid examples of operations.

    Args:
        operation (callable) : A callable that takes two streamlines dicts as
            inputs and preduces a new streamline dict.
        streamlines (list of list of streamlines) : The streamlines used in
            the operation.
        precision (int, optional) : The number of decimals to keep when
            hashing the points of the streamlines. Allows a soft
            comparison of streamlines. If None, no rounding is performed.

    Returns:
        The streamlines obtained after performing the operation on all the
            input streamlines.
        The indices of the streamlines that are used in the output.

    """

    # Hash the streamlines using the desired precision.
    indices = np.cumsum([0] + [len(s) for s in streamlines[:-1]])
    hashes = [hash_streamlines(s, i, precision) for
              s, i in zip(streamlines, indices)]

    # Perform the operation on the hashes and get the output streamlines.
    to_keep = reduce(operation, hashes)
    all_streamlines = list(itertools.chain(*streamlines))
    indices = sorted(to_keep.values())
    streamlines = [all_streamlines[i] for i in indices]
    return streamlines, indices


def union(left, right):
    """Union of two streamlines dict (see hash_streamlines)"""

    # In python 3 : return {**left, **right}
    result = left.copy()
    result.update(right)
    return result
