# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin

from scilpy.image.volume_space_management import DataVolume
from scilpy.tractograms.dps_and_dpp_management import (
    add_data_as_color_dpp, convert_dps_to_dpp, project_map_to_streamlines,
    project_dpp_to_map, perform_operation_on_dpp, perform_operation_dpp_to_dps,
    perform_correlation_on_endpoints)
from scilpy.viz.color import get_lookup_table


def _get_small_sft():
    # SFT = 2 streamlines: [3 points, 4 points]
    fake_ref = nib.Nifti1Image(np.zeros((3, 3, 3)), affine=np.eye(4))
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


def nan_array_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    nan_a = np.argwhere(np.isnan(a))
    nan_b = np.argwhere(np.isnan(a))

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    return np.array_equal(a, b) and np.array_equal(nan_a, nan_b)


def test_add_data_as_color_dpp():
    lut = get_lookup_table('viridis')

    # Important. cmap(1) != cmap(1.0)
    lowest_color = np.asarray(lut(0.0)[0:3]) * 255
    highest_color = np.asarray(lut(1.0)[0:3]) * 255

    fake_sft = _get_small_sft()

    # Not testing the clipping options. Will be tested through viz.utils tests

    # Test 1: One value per point.
    # Lowest cmap color should be first point of second streamline.
    some_data = [[2, 20, 200], [0.1, 0.3, 22, 5]]
    colored_sft, lbound, ubound = add_data_as_color_dpp(
        fake_sft, lut, some_data)
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
        fake_sft, lut, some_data)
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
    # All points of SFT are in voxel #0 or #1 in all dimensions.
    fake_sft = _get_small_sft()

    # Test 1. Verify on 3D volume = 1 value per point.
    map_data = np.zeros((3, 3, 3))
    map_data[0, 0, 0] = 1
    map_data[1, 1, 1] = 2
    map_volume = DataVolume(map_data, voxres=[1, 1, 1],
                            interpolation='nearest')

    # Test 1A. All points
    dpp = project_map_to_streamlines(fake_sft, map_volume)
    fake_sft.data_per_point['test1A'] = dpp  # Will fail if not the right shape
    assert np.array_equal(dpp[0].squeeze(), [1] * 3)
    assert np.array_equal(dpp[1].squeeze(), [2] * 4)

    # Test 1B. Endpoints
    dpp = project_map_to_streamlines(fake_sft, map_volume, endpoints_only=True)
    fake_sft.data_per_point['test1B'] = dpp  # Will fail if not the right shape
    assert nan_array_equal(dpp[0].squeeze(), [1, np.nan, 1.0])
    assert nan_array_equal(dpp[1].squeeze(), [2, np.nan, np.nan, 2.0])

    # -----------------

    # Test 2. Verify on 4D volume = N values per point.
    # (all points)
    map_data = np.zeros((3, 3, 3, 2))
    map_data[0, 0, 0, :] = 1
    map_data[1, 1, 1, :] = 2
    map_volume = DataVolume(map_data, voxres=[1, 1, 1],
                            interpolation='nearest')

    dpp = project_map_to_streamlines(fake_sft, map_volume)
    fake_sft.data_per_point['test2A'] = dpp  # Will fail if not the right shape
    assert np.array_equal(dpp[0], [[1, 1]] * 3)
    assert np.array_equal(dpp[1], [[2, 2]] * 4)


def test_project_dpp_to_map():
    fake_sft = _get_small_sft()
    fake_sft.data_per_point['my_dpp'] = [[1]*3, [2]*4]

    map_data = project_dpp_to_map(fake_sft, 'my_dpp', sum_lines=True)

    expected = np.zeros((3, 3, 3))  # fake_ref is 3x3x3
    expected[0, 0, 0] = 3 * 1  # the 3 points of the first streamline
    expected[1, 1, 1] = 4 * 2  # the 4 points of the second streamline
    assert np.array_equal(map_data, expected)


def test_perform_operation_on_dpp():
    fake_sft = _get_small_sft()
    fake_sft.data_per_point['my_dpp'] = [[[1, 0]]*3,
                                         [[2, 0]]*4]

    # Mean:
    dpp = perform_operation_on_dpp('mean', fake_sft, 'my_dpp')
    assert np.array_equal(dpp[0].squeeze(), [0.5] * 3)
    assert np.array_equal(dpp[1].squeeze(), [1] * 4)

    # Sum:
    dpp = perform_operation_on_dpp('sum', fake_sft, 'my_dpp')
    assert np.array_equal(dpp[0].squeeze(), [1] * 3)
    assert np.array_equal(dpp[1].squeeze(), [2] * 4)

    # Min:
    dpp = perform_operation_on_dpp('min', fake_sft, 'my_dpp')
    assert np.array_equal(dpp[0].squeeze(), [0] * 3)
    assert np.array_equal(dpp[1].squeeze(), [0] * 4)

    # Max:
    dpp = perform_operation_on_dpp('max', fake_sft, 'my_dpp')
    assert np.array_equal(dpp[0].squeeze(), [1] * 3)
    assert np.array_equal(dpp[1].squeeze(), [2] * 4)


def test_perform_operation_dpp_to_dps():
    fake_sft = _get_small_sft()
    fake_sft.data_per_point['my_dpp'] = [[[1, 0]]*3,
                                         [[2, 0]]*4]

    # Mean:
    dps = perform_operation_dpp_to_dps('mean', fake_sft, 'my_dpp')
    assert np.array_equal(dps[0], [1, 0])
    assert np.array_equal(dps[1], [2, 0])

    # Sum:
    dps = perform_operation_dpp_to_dps('sum', fake_sft, 'my_dpp')
    assert np.array_equal(dps[0], [3 * 1, 0])
    assert np.array_equal(dps[1], [4 * 2, 0])

    # Min:
    dps = perform_operation_dpp_to_dps('min', fake_sft, 'my_dpp')
    assert np.array_equal(dps[0], [1, 0])
    assert np.array_equal(dps[1], [2, 0])

    # Max:
    dps = perform_operation_dpp_to_dps('max', fake_sft, 'my_dpp')
    assert np.array_equal(dps[0], [1, 0])
    assert np.array_equal(dps[1], [2, 0])


def test_perform_correlation_on_endpoints():
    fake_sft = _get_small_sft()
    fake_sft.data_per_point['my_dpp'] = [[[1, 0]]*3,
                                         [[2, 0], [0, 0], [0, 0], [0, 0]]]
    dps = perform_correlation_on_endpoints(fake_sft, 'my_dpp')
    assert np.allclose(dps[0], [1])
    assert nan_array_equal(dps[1], [np.nan])
