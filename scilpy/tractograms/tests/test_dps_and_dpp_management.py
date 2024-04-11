# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin

from scilpy.tractograms.dps_and_dpp_management import add_data_as_color_dpp, \
    convert_dps_to_dpp
from scilpy.viz.utils import get_colormap

# SFT = 2 streamlines: [3 points, 4 points]
def _get_small_sft():
    fake_ref = nib.Nifti1Image(np.zeros((3, 3)), affine=np.eye(4))
    fake_sft = StatefulTractogram(streamlines=[[[0.1, 0.1, 0.1],
                                                [0.2, 0.2, 0.2],
                                                [0.3, 0.3, 0.3]],
                                               [[1.1, 1.1, 1.1],
                                                [1.2, 1.2, 1.2],
                                                [1.3, 1.3, 1.3],
                                                [1.4, 1.4, 1.4]]],
                                  reference=fake_ref,
                                  space=Space.VOX, origin=Origin('corner'))
    return fake_sft


def test_add_data_as_color_dpp():
    cmap = get_colormap('viridis')

    # Important. cmap(1) != cmap(1.0)
    lowest_color = np.asarray(cmap(0.0)[0:3]) * 255
    highest_color = np.asarray(cmap(1.0)[0:3]) * 255

    fake_sft = _get_small_sft()

    # Not testing the clipping options. Will be tested through viz.utils tests

    # Test 1: One value per point.
    # Lowest cmap color should be first point of second streamline.
    some_data = [[2, 20, 200], [0.1, 0.3, 22, 5]]
    colored_sft, lbound, ubound = add_data_as_color_dpp(
        fake_sft, cmap, some_data)
    assert len(colored_sft.data_per_streamline.keys()) == 0
    assert list(colored_sft.data_per_point.keys()) == ['color']
    assert lbound == 0.1
    assert ubound == 200
    assert np.array_equal(colored_sft.data_per_point['color'][1][0, :],
                          lowest_color)
    assert np.array_equal(colored_sft.data_per_point['color'][0][2, :],
                          highest_color)

    # Test 2: One value per streamline
    # Lowest cmap color should be every point in first streamline
    some_data = np.asarray([4, 5])
    colored_sft, lbound, ubound = add_data_as_color_dpp(
        fake_sft, cmap, some_data)
    assert len(colored_sft.data_per_streamline.keys()) == 0
    assert list(colored_sft.data_per_point.keys()) == ['color']
    assert lbound == 4
    assert ubound == 5
    # Lowest cmap color should be first point of second streamline.
    # Same value for all points.
    colors_first_line = colored_sft.data_per_point['color'][0]
    assert np.array_equal(colors_first_line[0, :], lowest_color)
    assert np.all(colors_first_line[1:, :] == colors_first_line[0, :])


def test_convert_dps_to_dpp():
    fake_sft = _get_small_sft()

    # Adding fake dps:
    fake_sft.data_per_streamline['my_dps'] = [5, 6]

    # Converting
    fake_sft = convert_dps_to_dpp(fake_sft, 'my_dps')
    assert len(fake_sft.data_per_streamline.keys()) == 0
    assert list(fake_sft.data_per_point.keys()) == ['my_dps']

    # Add again, will fail. Allow overwrite.
    fake_sft.data_per_streamline['my_dps'] = [5, 6]
    failed = False
    try:
        _ = convert_dps_to_dpp(fake_sft, 'my_dps')
    except ValueError:
        failed = True
        _ = convert_dps_to_dpp(fake_sft, 'my_dps', overwrite=True)

    assert failed


def test_project_map_to_streamlines():
    # toDo
    pass


def test_project_dpp_to_map():
    pass


def test_perform_operation_on_dpp():
    # toDo
    pass


def test_perform_operation_dpp_to_dps():
    # toDo
    pass


def test_pairwise_streamline_operation_on_endpoints():
    pass


def test_perform_streamline_operation_on_endpoints():
    # toDo
    pass
