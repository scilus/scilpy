# -*- coding: utf-8 -*-
import logging
import os
import tempfile

import numpy as np
from dipy.io.streamline import load_tractogram

from scilpy.io.fetcher import fetch_data, get_testing_files_dict, get_home
from scilpy.tractograms.tractogram_operations import flip_sft, \
    shuffle_streamlines, perform_tractogram_operation_on_lines, intersection, union, \
    difference, intersection_robust, difference_robust, union_robust, \
    concatenate_sft, perform_tractogram_operation_on_sft

# Prepare SFT
fetch_data(get_testing_files_dict(), keys='surface_vtk_fib.zip')
tmp_dir = tempfile.TemporaryDirectory()
in_sft = os.path.join(get_home(), 'surface_vtk_fib', 'gyri_fanning.trk')

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
    sft2 = sft.from_sft(sft.streamlines, sft)
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

