# -*- coding: utf-8 -*-
import os
import tempfile

from dipy.io.streamline import load_tractogram
from dipy.tracking.streamlinespeed import length
import nibabel as nib
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamlinespeed import length

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.tractograms.streamline_operations import (
    filter_streamlines_by_length,
    filter_streamlines_by_total_length_per_dim,
    resample_streamlines_num_points,
    resample_streamlines_step_size,
    smooth_line_gaussian,
    smooth_line_spline,
    parallel_transport_streamline)
from scilpy.tractograms.tractogram_operations import concatenate_sft


fetch_data(get_testing_files_dict(), keys=['tractograms.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def _setup_files():
    """ Load streamlines and masks relevant to the tests here.
    """

    os.chdir(os.path.expanduser(tmp_dir.name))
    in_long_sft = os.path.join(SCILPY_HOME, 'tractograms',
                               'streamline_operations',
                               'bundle_4.tck')
    in_mid_sft = os.path.join(SCILPY_HOME, 'tractograms',
                              'streamline_operations',
                              'bundle_4_cut_endpoints.tck')
    in_short_sft = os.path.join(SCILPY_HOME, 'tractograms',
                                'streamline_operations',
                                'bundle_4_cut_center.tck')
    in_ref = os.path.join(SCILPY_HOME, 'tractograms',
                          'streamline_operations',
                          'bundle_4_wm.nii.gz')

    in_rois = os.path.join(SCILPY_HOME, 'tractograms',
                           'streamline_operations',
                           'bundle_4_head_tail_offset.nii.gz')

    # Load sft
    long_sft = load_tractogram(in_long_sft, in_ref)
    mid_sft = load_tractogram(in_mid_sft, in_ref)
    short_sft = load_tractogram(in_short_sft, in_ref)

    sft = concatenate_sft([long_sft, mid_sft, short_sft])

    # Load mask
    rois = nib.load(in_rois)
    return sft, rois


def test_angles():
    # toDo
    pass


def test_get_values_along_length():
    # toDo
    pass


def test_compress_sft():
    # toDo
    pass


def test_cut_invalid_streamlines():
    # toDo
    pass


def test_filter_streamlines_by_length_max_length():
    """ Test the filter_streamlines_by_length function with a max length.
    """

    sft, _ = _setup_files()

    min_length = 0.
    max_length = 100
    # Filter streamlines by length and get the lengths
    filtered_sft = filter_streamlines_by_length(
        sft, min_length=min_length, max_length=max_length)
    lengths = length(filtered_sft.streamlines)

    # Test that streamlines were removed and that the test is not bogus.
    assert len(filtered_sft) < len(sft)

    assert np.all(lengths <= max_length)


def test_filter_streamlines_by_length_min_length():
    """ Test the filter_streamlines_by_length function with a min length.
    """

    sft, _ = _setup_files()

    min_length = 100
    max_length = np.inf

    # Filter streamlines by length and get the lengths
    filtered_sft = filter_streamlines_by_length(
        sft, min_length=min_length, max_length=max_length)
    lengths = length(filtered_sft.streamlines)

    # Test that streamlines were removed and that the test is not bogus.
    assert len(filtered_sft) < len(sft)
    # Test that streamlines shorter than 100 were removed.
    assert np.all(lengths >= min_length)


def test_filter_streamlines_by_length_min_and_max_length():
    """ Test the filter_streamlines_by_length function with a min
    and max length.
    """

    sft, _ = _setup_files()

    min_length = 100
    max_length = 120

    # Filter streamlines by length and get the lengths
    filtered_sft = filter_streamlines_by_length(
        sft, min_length=min_length, max_length=max_length)
    lengths = length(filtered_sft.streamlines)

    # Test that streamlines were removed and that the test is not bogus.
    assert len(filtered_sft) < len(sft)
    # Test that streamlines shorter than 100 and longer than 120 were removed.
    assert np.all(lengths >= min_length) and np.all(lengths <= max_length)


def test_filter_streamlines_by_total_length_per_dim_x():
    """ Test the filter_streamlines_by_total_length_per_dim function.
    This function is quite awkward to test without reimplementing
    the logic, but luckily we have data going purely left-right.

    This test also tests the return of rejected streamlines.
    """

    # Streamlines are going purely left-right, so the
    # x dimension should have the longest span.
    sft, _ = _setup_files()

    min_length = 115
    max_length = 125

    constraint = [min_length, max_length]
    inf_constraint = [-np.inf, np.inf]

    # Filter streamlines by length and get the lengths
    # No rejected streamlines should be returned
    filtered_sft, ids, rejected = filter_streamlines_by_total_length_per_dim(
        sft, constraint, inf_constraint, inf_constraint,
        True, False)
    lengths = length(filtered_sft.streamlines)

    # Test that streamlines were removed and that the test is not bogus.
    assert len(filtered_sft) < len(sft)
    # Remaining streamlines should have the correct length
    assert np.all(lengths >= min_length) and np.all(lengths <= max_length)
    # No rejected streamlines should have been returned
    assert rejected is None


def test_filter_streamlines_by_total_length_per_dim_y():
    """ Test the filter_streamlines_by_total_length_per_dim function.
    This function is quite awkward to test without reimplementing
    the logic. We rotate the streamlines to be purely up-down.

    This test also tests the return of rejected streamlines. The rejected
    streamlines should have "invalid" lengths.
    """

    # Streamlines are going purely left-right, so the
    # streamlines have to be rotated to be purely up-down.
    sft, _ = _setup_files()

    # Rotate streamlines by swapping x and y for all streamlines
    swapped_streamlines_y = [s[:, [1, 0, 2]] for s in sft.streamlines]
    sft_y = sft.from_sft(swapped_streamlines_y, sft)

    min_length = 115
    max_length = 125

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


def test_filter_streamlines_by_total_length_per_dim_z():
    """ Test the filter_streamlines_by_total_length_per_dim function.
    This function is quite awkward to test without reimplementing
    the logic.
    """

    # Streamlines are going purely left-right, so the
    # streamlines have to be rotated to be purely front-back.
    sft, _ = _setup_files()

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


def test_resample_streamlines_num_points_2():
    """ Test the resample_streamlines_num_points function to 2 points.
    """

    sft, _ = _setup_files()
    nb_points = 2

    resampled_sft = resample_streamlines_num_points(sft, nb_points)
    lengths = [len(s) == nb_points for s in resampled_sft.streamlines]

    assert np.all(lengths)


def test_resample_streamlines_num_points_1000():
    """ Test the resample_streamlines_num_points function to 1000 points.
    """

    sft, _ = _setup_files()
    nb_points = 1000

    resampled_sft = resample_streamlines_num_points(sft, nb_points)
    lengths = [len(s) == nb_points for s in resampled_sft.streamlines]

    assert np.all(lengths)


def test_resample_streamlines_step_size_1mm():
    """ Test the resample_streamlines_step_size function to 1mm.
    """

    sft, _ = _setup_files()

    step_size = 1.0
    resampled_sft = resample_streamlines_step_size(sft, step_size)

    # Compute the step size of each streamline and concatenate them
    # to get a single array of steps
    steps = np.concatenate([np.linalg.norm(np.diff(s, axis=0), axis=-1)
                            for s in resampled_sft.streamlines])
    # Tolerance of 10% of the step size
    assert np.allclose(steps, step_size, atol=0.1), steps


def test_resample_streamlines_step_size_01mm():
    """ Test the resample_streamlines_step_size function to 0.1mm.
    """

    sft, _ = _setup_files()

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

    sft, _ = _setup_files()
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

    sft, _ = _setup_files()
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

    sft, _ = _setup_files()
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

    sft, _ = _setup_files()
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
    sft, _ = _setup_files()
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
