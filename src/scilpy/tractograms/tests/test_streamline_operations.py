# -*- coding: utf-8 -*-
import os
import tempfile

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamlinespeed import length
from dipy.io.stateful_tractogram import StatefulTractogram

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.tractograms.streamline_operations import (
    compress_sft,
    cut_invalid_streamlines,
    filter_streamlines_by_length,
    filter_streamlines_by_total_length_per_dim,
    get_angles,
    get_streamlines_as_linspaces,
    resample_streamlines_num_points,
    resample_streamlines_step_size,
    smooth_line_gaussian,
    smooth_line_spline,
    parallel_transport_streamline,
    remove_overlapping_points_streamlines,
    filter_streamlines_by_nb_points)
from scilpy.tractograms.tractogram_operations import concatenate_sft

fetch_data(get_testing_files_dict(), keys=['tractograms.zip'])
tmp_dir = tempfile.TemporaryDirectory()

# Streamlines and masks relevant to the tests here.
test_files_path = os.path.join(SCILPY_HOME, 'tractograms',
                               'streamline_operations')
in_long_sft = os.path.join(test_files_path, 'bundle_4.tck')
in_mid_sft = os.path.join(test_files_path, 'bundle_4_cut_endpoints.tck')
in_short_sft = os.path.join(test_files_path, 'bundle_4_cut_center.tck')
in_ref = os.path.join(test_files_path, 'bundle_4_wm.nii.gz')
in_rois = os.path.join(test_files_path, 'bundle_4_head_tail_offset.nii.gz')


def test_get_angles():
    fake_straight_line = np.asarray([[0, 0, 0],
                                     [1, 1, 1],
                                     [2, 2, 2],
                                     [3, 3, 3]], dtype=float)
    fake_ninety_degree = np.asarray([[0, 0, 0],
                                     [1, 1, 0],
                                     [0, 2, 0]], dtype=float)

    sft = load_tractogram(in_short_sft, in_ref)
    sft.streamlines = [fake_straight_line, fake_ninety_degree]

    angles = get_angles(sft)
    assert np.array_equal(angles[0], [0, 0])
    assert np.array_equal(angles[1], [90])

    angles = get_angles(sft, add_zeros=True)
    assert np.array_equal(angles[0], [0, 0, 0, 0])
    assert np.array_equal(angles[1], [0, 90, 0])


def test_get_streamlines_as_linspaces():
    sft = load_tractogram(in_short_sft, in_ref)
    lines = get_streamlines_as_linspaces(sft)
    assert len(lines) == len(sft)
    assert len(lines[0]) == len(sft.streamlines[0])
    assert lines[0][0] == 0
    assert lines[0][-1] == 1


def test_compress_sft():
    sft = load_tractogram(in_long_sft, in_ref)
    compressed = compress_sft(sft, tol_error=0.01)
    assert len(sft) == len(compressed)

    for s, sc in zip(sft.streamlines, compressed.streamlines):
        # All streamlines should be shorter once compressed
        assert len(sc) <= len(s)

        # Streamlines' endpoints should not be changed
        assert np.allclose(s[0], sc[0])
        assert np.allclose(s[-1], sc[-1])

        # Not testing more than that, as it uses Dipy's method, tested by Dipy


def test_cut_invalid_streamlines():
    sft = load_tractogram(in_short_sft, in_ref)
    sft.to_vox()

    cut, nb = cut_invalid_streamlines(sft)
    assert len(cut) == len(sft)
    assert nb == 0

    # Faking an invalid streamline at all positions.
    # Currently, volume is 64x64x3
    remaining_streamlines = [11, 10, 9, 8, 7, 6, 6, 7, 8, 9, 10, 11]
    for index, ind_cut in enumerate(sft.streamlines[0]):
        sft = load_tractogram(in_short_sft, in_ref)
        sft.streamlines[0][index, :] = [65.0, 65.0, 2.0]
        cut, nb = cut_invalid_streamlines(sft)
        assert len(cut) == len(sft)
        assert np.all([len(sc) <= len(s) for s, sc in
                       zip(sft.streamlines, cut.streamlines)])
        assert len(cut.streamlines[0]) == remaining_streamlines[index]
        assert nb == 1

    # Faking an invalid streamline at position 0 and -1
    sft = load_tractogram(in_short_sft, in_ref)
    sft.streamlines[0][0, :] = [65.0, 65.0, 2.0]
    sft.streamlines[0][-1, :] = [65.0, 65.0, 2.0]
    cut, nb = cut_invalid_streamlines(sft)
    assert len(cut) == len(sft)
    assert np.all([len(sc) <= len(s) for s, sc in
                   zip(sft.streamlines, cut.streamlines)])
    assert len(cut.streamlines[0]) == len(sft.streamlines[0]) - 2
    assert nb == 1

    # Faking an invalid streamline at position 2 and 2
    sft = load_tractogram(in_short_sft, in_ref)
    sft.streamlines[0][2, :] = [65.0, 65.0, 2.0]
    sft.streamlines[0][9, :] = [65.0, 65.0, 2.0]
    cut, nb = cut_invalid_streamlines(sft)
    assert len(cut) == len(sft)
    assert np.all([len(sc) <= len(s) for s, sc in
                   zip(sft.streamlines, cut.streamlines)])
    assert len(cut.streamlines[0]) == 6
    assert nb == 1


def test_filter_streamlines_by_min_nb_points_2():
    sft = load_tractogram(in_short_sft, in_ref)

    # Adding a one-point streamline
    sft.streamlines.append([[7, 7, 7]])
    new_sft = filter_streamlines_by_nb_points(sft, min_nb_points=2)
    assert len(new_sft) == len(sft) - 1


def test_filter_streamlines_min_by_nb_points_5():
    sft = load_tractogram(in_short_sft, in_ref)

    # Adding a one-point streamline
    sft.streamlines.append([[7, 7, 7],
                            [7, 7, 7],
                            [7, 7, 7],
                            [7, 7, 7],
                            [7, 7, 7], ])

    sft.streamlines.append([[7, 7, 7],
                            [7, 7, 7],
                            [7, 7, 7],
                            [7, 7, 7], ])

    new_sft = filter_streamlines_by_nb_points(sft, min_nb_points=5)
    assert len(new_sft) == len(sft) - 1


def test_remove_overlapping_points_streamlines():
    sft = load_tractogram(in_short_sft, in_ref)

    fake_line = np.asarray([[3, 3, 3],
                            [4, 4, 4],
                            [5, 5, 5],
                            [5, 5, 5.00000001]], dtype=float)
    sft.streamlines.append(fake_line)

    new_sft = remove_overlapping_points_streamlines(sft)
    assert len(new_sft.streamlines[-1]) == len(sft.streamlines[-1]) - 1
    assert np.all([len(new_sft.streamlines[i]) == len(sft.streamlines[i]) for
                   i in range(len(sft) - 1)])


def test_filter_streamlines_by_length():
    long_sft = load_tractogram(in_long_sft, in_ref)
    mid_sft = load_tractogram(in_mid_sft, in_ref)
    short_sft = load_tractogram(in_short_sft, in_ref)
    sft = concatenate_sft([long_sft, mid_sft, short_sft])

    # === 1. Using max length ===
    min_length = 0.
    max_length = 100
    # Filter streamlines by length and get the lengths
    filtered_sft, _ = filter_streamlines_by_length(
        sft, min_length=min_length, max_length=max_length)
    lengths = length(filtered_sft.streamlines)

    # Test that streamlines were removed and that the test is not bogus.
    assert len(filtered_sft) < len(sft)

    assert np.all(lengths <= max_length)

    # === 2. Using min length ===
    min_length = 100
    max_length = np.inf

    # Filter streamlines by length and get the lengths
    filtered_sft, _ = filter_streamlines_by_length(
        sft, min_length=min_length, max_length=max_length)
    lengths = length(filtered_sft.streamlines)

    # Test that streamlines were removed and that the test is not bogus.
    assert len(filtered_sft) < len(sft)
    # Test that streamlines shorter than 100 were removed.
    assert np.all(lengths >= min_length)

    # === 3. Using both min and max length ===
    min_length = 100
    max_length = 120

    # Filter streamlines by length and get the lengths
    filtered_sft, _ = filter_streamlines_by_length(
        sft, min_length=min_length, max_length=max_length)
    lengths = length(filtered_sft.streamlines)

    # Test that streamlines were removed and that the test is not bogus.
    assert len(filtered_sft) < len(sft)
    # Test that streamlines shorter than 100 and longer than 120 were removed.
    assert np.all(lengths >= min_length) and np.all(lengths <= max_length)

    # === 4. Return rejected streamlines with empty sft ===
    empty_sft = short_sft[[]]  # Empty sft from short_sft (chosen arbitrarily)
    filtered_sft, _, rejected = \
        filter_streamlines_by_length(empty_sft, min_length=min_length,
                                     max_length=max_length,
                                     return_rejected=True)
    assert isinstance(filtered_sft, StatefulTractogram)
    assert isinstance(rejected, StatefulTractogram)
    assert len(filtered_sft) == 0
    assert len(rejected) == 0


def test_filter_streamlines_by_total_length_per_dim():
    long_sft = load_tractogram(in_long_sft, in_ref)
    mid_sft = load_tractogram(in_mid_sft, in_ref)
    short_sft = load_tractogram(in_short_sft, in_ref)
    sft = concatenate_sft([long_sft, mid_sft, short_sft])

    min_length = 115
    max_length = 125

    # === 1. Test x dimension ===
    # Test sft has streamlines that are going purely left-right, so the x
    # dimension should have the longest span.
    constraint = [min_length, max_length]
    inf_constraint = [-np.inf, np.inf]

    # Filter streamlines by length and get the lengths
    # No rejected streamlines should be returned
    filtered_sft, ids, rejected = filter_streamlines_by_total_length_per_dim(
        sft, constraint, inf_constraint, inf_constraint,
        use_abs=True, save_rejected=False)
    lengths = length(filtered_sft.streamlines)

    # Test that streamlines were removed and that the test is not bogus.
    assert len(filtered_sft) < len(sft)
    # Remaining streamlines should have the correct length
    assert np.all(lengths >= min_length) and np.all(lengths <= max_length)
    # No rejected streamlines should have been returned
    assert rejected is None

    # === 2. Testing y dimension ===

    # Rotate streamlines by swapping x and y for all streamlines
    swapped_streamlines_y = [s[:, [1, 0, 2]] for s in sft.streamlines]
    sft_y = sft.from_sft(swapped_streamlines_y, sft)

    constraint = [min_length, max_length]
    inf_constraint = [-np.inf, np.inf]

    # Filter streamlines by length and get the lengths
    filtered_sft, _, rejected = filter_streamlines_by_total_length_per_dim(
        sft_y, inf_constraint, constraint, inf_constraint,
        True, True)
    lengths = length(filtered_sft.streamlines)
    rejected_lengths = length(rejected.streamlines)

    # Test that streamlines were removed and that the test is not bogus.
    assert len(filtered_sft) < len(sft)
    assert np.all(lengths >= min_length) and np.all(lengths <= max_length)
    assert np.all(np.logical_or(min_length > rejected_lengths,
                                rejected_lengths > max_length))

    # === 3. Testing z dimension ===
    # Rotate streamlines by swapping x and z for all streamlines
    swapped_streamlines_y = [s[:, [2, 1, 0]] for s in sft.streamlines]
    sft_y = sft.from_sft(swapped_streamlines_y, sft)

    min_length = 115
    max_length = 125

    constraint = [min_length, max_length]
    inf_constraint = [-np.inf, np.inf]

    # Filter streamlines by length and get the lengths
    filtered_sft, _, _ = filter_streamlines_by_total_length_per_dim(
        sft_y, inf_constraint, inf_constraint, constraint,
        True, False)
    lengths = length(filtered_sft.streamlines)

    # Test that streamlines were removed and that the test is not bogus.
    assert len(filtered_sft) < len(sft)
    assert np.all(lengths >= min_length) and np.all(lengths <= max_length)


def test_resample_streamlines_num_points():
    long_sft = load_tractogram(in_long_sft, in_ref)
    mid_sft = load_tractogram(in_mid_sft, in_ref)
    short_sft = load_tractogram(in_short_sft, in_ref)
    sft = concatenate_sft([long_sft, mid_sft, short_sft])

    # Test 1. To two points
    nb_points = 2
    resampled_sft = resample_streamlines_num_points(sft, nb_points)
    lengths = [len(s) == nb_points for s in resampled_sft.streamlines]
    assert np.all(lengths)

    # Test 2. To 1000 points.
    nb_points = 1000
    resampled_sft = resample_streamlines_num_points(sft, nb_points)
    lengths = [len(s) == nb_points for s in resampled_sft.streamlines]

    assert np.all(lengths)


def test_resample_streamlines_step_size():
    """ Test the resample_streamlines_step_size function to 1mm.
    """
    long_sft = load_tractogram(in_long_sft, in_ref)
    mid_sft = load_tractogram(in_mid_sft, in_ref)
    short_sft = load_tractogram(in_short_sft, in_ref)
    sft = concatenate_sft([long_sft, mid_sft, short_sft])

    # Test 1. To 1 mm
    step_size = 1.0
    resampled_sft = resample_streamlines_step_size(sft, step_size)

    # Compute the step size of each streamline and concatenate them
    # to get a single array of steps
    steps = np.concatenate([np.linalg.norm(np.diff(s, axis=0), axis=-1)
                            for s in resampled_sft.streamlines])
    # Tolerance of 10% of the step size
    assert np.allclose(steps, step_size, atol=0.1), steps

    # Test 2. To 0.1 mm
    step_size = 0.1
    resampled_sft = resample_streamlines_step_size(sft, step_size)

    # Compute the step size of each streamline and concatenate them
    # to get a single array of steps
    steps = np.concatenate([np.linalg.norm(np.diff(s, axis=0), axis=-1)
                            for s in resampled_sft.streamlines])
    # Tolerance of 10% of the step size
    assert np.allclose(steps, step_size, atol=0.01), steps


def test_smooth_line_gaussian_error():
    """ Test the smooth_line_gaussian function by adding noise to a
    streamline and smoothing it. The function does not accept a sigma
    value of 0, therefore it should throw and error.
    """

    sft = load_tractogram(in_long_sft, in_ref)
    streamline = sft.streamlines[0]

    # Add noise to the streamline
    noisy_streamline = streamline + np.random.normal(0, 0.1, streamline.shape)

    # Should throw a ValueError
    with pytest.raises(ValueError):
        _ = smooth_line_gaussian(noisy_streamline, 0.0)


def test_smooth_line_gaussian():
    """ Test the smooth_line_gaussian function by adding noise to a
    streamline and smoothing it. The smoothed streamline should be
    closer to the original streamline than the noisy one.
    """
    sft = load_tractogram(in_long_sft, in_ref)
    streamline = sft.streamlines[0]

    rng = np.random.default_rng(1337)

    # Add noise to the streamline
    noise = rng.normal(0, 0.5, streamline.shape)
    noise[0] = np.zeros(3)
    noise[-1] = np.zeros(3)
    noisy_streamline = streamline + noise

    # Smooth the noisy streamline
    smoothed_streamline = smooth_line_gaussian(noisy_streamline, 5.0)

    # Compute the distance between the original and smoothed streamline
    # and between the noisy and smoothed streamline
    dist_1 = np.linalg.norm(streamline - smoothed_streamline)
    dist_2 = np.linalg.norm(noisy_streamline - smoothed_streamline)

    assert dist_1 < dist_2


def test_smooth_line_spline_error():
    """ Test the smooth_line_spline function by adding noise to a
    streamline and smoothing it. The function does not accept a sigma
    value of 0, therefore it should throw and error.
    """
    sft = load_tractogram(in_long_sft, in_ref)
    streamline = sft.streamlines[0]

    # Add noise to the streamline
    noisy_streamline = streamline + np.random.normal(0, 0.1, streamline.shape)

    # Should throw a ValueError
    with pytest.raises(ValueError):
        _ = smooth_line_spline(noisy_streamline, 0.0, 10)


def test_smooth_line_spline():
    """ Test the smooth_line_spline function by adding noise to a
    streamline and smoothing it. The smoothed streamline should be
    closer to the original streamline than the noisy one.
    """
    sft = load_tractogram(in_short_sft, in_ref)
    streamline = sft.streamlines[-1]

    rng = np.random.default_rng(1337)

    # Add noise to the streamline
    noise = rng.normal(0, 0.5, streamline.shape)
    noise[0] = np.zeros(3)
    noise[-1] = np.zeros(3)
    noisy_streamline = streamline + noise

    # Smooth the noisy streamline
    smoothed_streamline = smooth_line_spline(noisy_streamline, 5., 10)

    # Compute the distance between the original and smoothed streamline
    # and between the noisy and smoothed streamline
    dist_1 = np.linalg.norm(streamline - smoothed_streamline)
    dist_2 = np.linalg.norm(noisy_streamline - smoothed_streamline)

    assert dist_1 < dist_2


def test_generate_matched_points():
    # toDo
    pass


def test_parallel_transport_streamline():
    sft = load_tractogram(in_long_sft, in_ref)
    streamline = sft.streamlines[0]

    rng = np.random.default_rng(3018)
    pt_streamlines = parallel_transport_streamline(
        streamline, 20, 5, rng)

    avg_streamline = np.mean(pt_streamlines, axis=0)

    assert_array_almost_equal(avg_streamline[0],
                              [-26.999582, -116.320145, 6.3678055],
                              decimal=4)
    assert_array_almost_equal(avg_streamline[-1],
                              [-155.99944, -116.56515, 6.2451267],
                              decimal=4)
    assert [len(s) for s in pt_streamlines] == [130] * 20
    assert len(pt_streamlines) == 20


def test_remove_loops():
    # toDO
    # Coverage will not work: uses multi-processing
    pass


def test_remove_sharp_turns_qb():
    # toDO
    pass


def test_remove_loops_and_sharp_turns():
    # ok. Just a combination of the two previous functions.
    pass
