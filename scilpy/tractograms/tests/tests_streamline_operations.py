# -*- coding: utf-8 -*-
import os
import tempfile

import numpy as np
from dipy.io.streamline import load_tractogram

from scilpy.io.fetcher import fetch_data, get_testing_files_dict, get_home
from scilpy.tractograms.streamline_operations import \
    resample_streamlines_num_points, resample_streamlines_step_size

# Prepare SFT
fetch_data(get_testing_files_dict(), keys='surface_vtk_fib.zip')
tmp_dir = tempfile.TemporaryDirectory()
in_sft = os.path.join(get_home(), 'surface_vtk_fib', 'gyri_fanning.trk')

# Loading and keeping only a few streamlines for faster testing.
sft = load_tractogram(in_sft, 'same')[0:4]


def test_filter_streamlines_by_length():
    # toDo
    pass


def test_filter_streamlines_by_total_length_per_dim():
    # toDo
    pass


def test_resample_streamlines_num_points():
    lengths = [len(s) for s in sft.streamlines]

    nb_points_down = min(lengths) - 1  # Downsampling all
    nb_points_up = max(lengths) + 1  # Upsampling all

    for nb_points in [nb_points_up, nb_points_down]:
        sft2 = resample_streamlines_num_points(sft, nb_points)
        lengths2 = [len(s) for s in sft2.streamlines]
        assert np.all(np.asarray(lengths2, dtype=int) == nb_points)


def test_resample_streamlines_step_size():
    step_size = 1.0
    sft2 = resample_streamlines_step_size(sft, step_size)

    # Checking only first streamline:
    steps = np.sqrt(np.sum(np.diff(sft2.streamlines[0], axis=0)**2, axis=-1))
    print(steps)

    # From our tests:
    #  - with step_size 0.5: steps are ~0.5079
    #  - with step_size 1.0: steps are ~1.048
    #  - with step_size 1.0: steps are ~2.24
    assert np.allclose(steps, step_size, atol=0.05)


def compute_streamline_segment():
    # toDo
    pass
