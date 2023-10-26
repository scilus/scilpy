# -*- coding: utf-8 -*-
import os
import tempfile

import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamlinespeed import length

from scilpy.io.fetcher import fetch_data, get_testing_files_dict, get_home
from scilpy.tractograms.streamline_operations import (
    filter_streamlines_by_length,
    resample_streamlines_num_points,
    resample_streamlines_step_size)

fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def _setup_files():
    """ Load streamlines and masks relevant to the tests here.
    """
    os.chdir(os.path.expanduser(tmp_dir.name))

    in_sft = os.path.join(get_home(), 'tracking',
                          'pft.trk')
    # Load sft
    sft = load_tractogram(in_sft, 'same')
    return sft


def test_filter_streamlines_by_length_max_length():
    """ Test the filter_streamlines_by_length function with a max length.
    """

    sft = _setup_files()

    min_length = 0.
    max_length = 100
    # Filter streamlines by length and get the lengths
    resampled_sft = filter_streamlines_by_length(
        sft, min_length=min_length, max_length=max_length)
    lengths = length(resampled_sft.streamlines)

    assert np.all(lengths <= max_length)


def test_filter_streamlines_by_length_min_length():
    """ Test the filter_streamlines_by_length function with a min length.
    """

    sft = _setup_files()

    min_length = 100
    max_length = np.inf

    # Filter streamlines by length and get the lengths
    resampled_sft = filter_streamlines_by_length(
        sft, min_length=min_length, max_length=max_length)
    lengths = length(resampled_sft.streamlines)

    assert np.all(lengths >= min_length)


def test_filter_streamlines_by_total_length_per_dim():
    # toDo
    pass


def test_resample_streamlines_num_points_2():
    """ Test the resample_streamlines_num_points function to 2 points.
    """

    sft = _setup_files()
    nb_points = 2

    resampled_sft = resample_streamlines_num_points(sft, nb_points)
    lengths = [len(s) == nb_points for s in resampled_sft.streamlines]

    assert np.all(lengths)


def test_resample_streamlines_num_points_1000():
    """ Test the resample_streamlines_num_points function to 1000 points.
    """

    sft = _setup_files()
    nb_points = 1000

    resampled_sft = resample_streamlines_num_points(sft, nb_points)
    lengths = [len(s) == nb_points for s in resampled_sft.streamlines]

    assert np.all(lengths)


def test_resample_streamlines_step_size_1mm():
    """ Test the resample_streamlines_step_size function to 1mm.
    """

    sft = _setup_files()

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

    sft = _setup_files()

    step_size = 0.1
    resampled_sft = resample_streamlines_step_size(sft, step_size)

    # Compute the step size of each streamline and concatenate them
    # to get a single array of steps
    steps = np.concatenate([np.linalg.norm(np.diff(s, axis=0), axis=-1)
                            for s in resampled_sft.streamlines])
    # Tolerance of 10% of the step size
    assert np.allclose(steps, step_size, atol=0.01), steps


def compute_streamline_segment():
    # toDo
    pass
