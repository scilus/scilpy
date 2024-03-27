# -*- coding: utf-8 -*-

import logging
import os
import tempfile

import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import load_tractogram

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.tractograms.tractogram_operations import (
    concatenate_sft,
    difference,
    difference_robust,
    flip_sft,
    intersection,
    intersection_robust,
    perform_tractogram_operation_on_lines,
    perform_tractogram_operation_on_sft,
    shuffle_streamlines,
    split_sft_randomly,
    split_sft_randomly_per_cluster,
    upsample_tractogram,
    union,
    union_robust)


# Prepare SFT
fetch_data(get_testing_files_dict(), keys=['surface_vtk_fib.zip'])
tmp_dir = tempfile.TemporaryDirectory()
in_sft = os.path.join(SCILPY_HOME, 'surface_vtk_fib', 'gyri_fanning.trk')

# Loading and keeping only a few streamlines for faster testing.
sft = load_tractogram(in_sft, 'same')[0:4]

# Faking data_per_streamline
sft.data_per_streamline['test'] = [1] * len(sft)
sft.data_per_point['test2'] = [[[1, 2, 3]] * len(s) for s in sft.streamlines]


def test_shuffle_streamlines():
    # Shuffling pretty straightforward, not testing.
    # Verifying that initial SFT is not modified.
    sft2 = shuffle_streamlines(sft)
    assert not sft2 == sft


def test_flip_sft():
    # Flip x, verify that y and z are the same.
    x, y, z = sft.streamlines[0][0]
    sft2 = flip_sft(sft, ['x'])
    x2, y2, z2 = sft2.streamlines[0][0]
    assert (not x == x2) and y == y2 and z == z2

    # Flip x and y, verify that z is the same.
    sft2 = flip_sft(sft, ['x', 'y'])
    x2, y2, z2 = sft2.streamlines[0][0]
    assert (not x == x2) and (not y == y2) and z == z2


def test_operations():
    same = sft.streamlines[0]
    different = np.asarray([[1., 0., 0.],
                            [1., 0., 0.],
                            [1., 0., 0.]])
    similar = same + 0.0001
    compared = [same, different, similar]

    # Intersection: should find 2 similar.
    output, indices = perform_tractogram_operation_on_lines(
        intersection, [[same], compared])
    assert len(output) == 1

    # Intersection less precise. Should find 3 similar.
    # (but can't be tested now; returns the rounded unique streamline)
    output, indices = perform_tractogram_operation_on_lines(
        intersection, [[same], compared], precision=1)
    assert len(output) == 1

    # Difference: A - B: should return 0
    output, indices = perform_tractogram_operation_on_lines(
        difference, [[same], compared])
    assert len(output) == 0

    # Difference: B - A: should return 2
    output, indices = perform_tractogram_operation_on_lines(
        difference, [compared, [same]])
    assert len(output) == 2

    # Difference: B - A less precise: should return 1
    output, indices = perform_tractogram_operation_on_lines(
        difference, [compared, [same]], precision=1)
    assert len(output) == 1

    # Union: should combine the two same,
    output, indices = perform_tractogram_operation_on_lines(
        union, [[same], compared])
    assert len(output) == 3

    # Union less precise: should combine the similar too
    output, indices = perform_tractogram_operation_on_lines(
        union, [[same], compared], precision=1)
    assert len(output) == 2


def test_robust_operations():

    # Recommended in scil_tractogram_math: use precision 0 to manage shifted
    # tractograms. Testing here.
    precision_shifted = 0

    same = sft.streamlines[0]
    shifted_same = same.copy() + 0.5
    different = np.asarray([[1., 0., 0.],
                            [1., 0., 0.],
                            [1., 0., 0.]])
    compared = [same, shifted_same, different]

    # Intersection: same/shifted
    output, indices = perform_tractogram_operation_on_lines(
        intersection_robust, [[same], compared], precision=precision_shifted)
    assert np.array_equal(indices, [0])
    assert len(output) == 1

    # Difference: different
    output, indices = perform_tractogram_operation_on_lines(
        difference_robust, [compared, [same]], precision=precision_shifted)
    logging.warning(indices)
    assert np.array_equal(indices, [2])
    assert len(output) == 1

    # Union: 4 (different, similar/same/shited)
    output, indices = perform_tractogram_operation_on_lines(
        union_robust, [compared, [same]], precision=precision_shifted)
    logging.warning(indices)
    assert len(output) == 2
    assert (indices == [0, 2]).all()


def test_concatenate_sft():
    # Testing with different metadata
    sft2 = StatefulTractogram.from_sft(sft.streamlines, sft)
    sft2.data_per_point['test2_different'] = [[['a', 'b', 'c']] * len(s)
                                              for s in sft.streamlines]

    failed = False
    try:
        total = concatenate_sft([sft, sft2])
    except ValueError:
        failed = True
    assert failed

    total = concatenate_sft([sft, sft])
    assert len(total) == len(sft) * 2
    assert len(total.data_per_streamline['test']) == 2 * len(sft)
    assert len(total.data_per_point['test2']) == 2 * len(sft)


def test_combining_sft():
    # todo
    perform_tractogram_operation_on_sft('union', [sft, sft], precision=None,
                                        fake_metadata=False, no_metadata=False)


def test_upsample_tractogram():
    new_sft = upsample_tractogram(sft, 1000, 0.5, 5, False, 0.1, 0)
    first_chunk = [[112.64021, 35.409477, 59.42175],
                   [109.09777, 35.287857, 61.845505],
                   [110.41855, 37.077374, 56.930523]]
    last_chunk = [[110.40285, 51.036686, 62.419273],
                  [109.698586, 48.330017, 64.50656],
                  [113.04737, 45.89119, 64.778534]]

    assert len(new_sft) == 1000
    assert len(new_sft.streamlines._data) == 8404
    assert np.allclose(first_chunk, new_sft.streamlines._data[0:30:10])
    assert np.allclose(last_chunk, new_sft.streamlines._data[-1:-31:-10])


def test_split_sft_randomly():
    sft_copy = StatefulTractogram.from_sft(sft.streamlines, sft)
    new_sft_list = split_sft_randomly(sft_copy, 2, 0)

    assert len(new_sft_list) == 2 and isinstance(new_sft_list, list)
    assert len(new_sft_list[0]) == 2 and len(new_sft_list[1]) == 2
    assert np.allclose(new_sft_list[0].streamlines[0][0],
                       [112.458, 35.7144, 58.7432])
    assert np.allclose(new_sft_list[1].streamlines[0][0],
                       [112.168, 35.259, 59.419])


def test_split_sft_randomly_per_cluster():
    sft_copy = StatefulTractogram.from_sft(sft.streamlines, sft)
    new_sft_list = split_sft_randomly_per_cluster(sft_copy, [2], 0,
                                                  [40, 30, 20, 10])
    assert len(new_sft_list) == 2 and isinstance(new_sft_list, list)
    assert len(new_sft_list[0]) == 2 and len(new_sft_list[1]) == 2
    assert np.allclose(new_sft_list[0].streamlines[0][0],
                       [112.168, 35.259, 59.419])
    assert np.allclose(new_sft_list[1].streamlines[0][0],
                       [112.266, 35.4188, 59.0421])


def filter_tractogram_data():
    # toDo
    pass
