# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import pytest

from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin

from scilpy.image.volume_space_management import DataVolume
from scilpy.tests.utils import nan_array_equal
from scilpy.tractograms.dps_and_dpp_management import (
    get_data_as_arraysequence,
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


def test_get_data_as_arraysequence_dpp():
    fake_sft = _get_small_sft()

    some_data = np.asarray([2, 20, 200, 0.1, 0.3, 22, 5])

    # Test 1: One value per point.
    array_seq = get_data_as_arraysequence(some_data, fake_sft)

    assert fake_sft._get_point_count() == array_seq.total_nb_rows


def test_get_data_as_arraysequence_dps():
    fake_sft = _get_small_sft()

    some_data = np.asarray([2, 20])

    # Test: One value per streamline.
    array_seq = get_data_as_arraysequence(some_data, fake_sft)
    assert fake_sft._get_streamline_count() == array_seq.total_nb_rows


def test_get_data_as_arraysequence_dps_2D():
    fake_sft = _get_small_sft()

    some_data = np.asarray([[2], [20]])

    # Test: One value per streamline.
    array_seq = get_data_as_arraysequence(some_data, fake_sft)
    assert fake_sft._get_streamline_count() == array_seq.total_nb_rows


def test_get_data_as_arraysequence_error():
    fake_sft = _get_small_sft()

    some_data = np.asarray([2, 20, 200, 0.1])

    # Test: Too many values per streamline, not enough per point.
    with pytest.raises(ValueError):
        _ = get_data_as_arraysequence(some_data, fake_sft)


def test_add_data_as_dpp_1_per_point():

    fake_sft = _get_small_sft()
    cmap = get_lookup_table('jet')

    # Test: One value per point.
    values = np.asarray([2, 20, 200, 0.1, 0.3, 22, 5])
    color = (np.asarray(cmap(values)[:, 0:3]) * 255).astype(np.uint8)

    array_seq = get_data_as_arraysequence(color, fake_sft)
    colored_sft = add_data_as_color_dpp(
        fake_sft, array_seq)
    assert len(colored_sft.data_per_streamline.keys()) == 0
    assert list(colored_sft.data_per_point.keys()) == ['color']


def test_add_data_as_dpp_1_per_streamline():

    fake_sft = _get_small_sft()
    cmap = get_lookup_table('jet')

    # Test: One value per streamline
    values = np.asarray([4, 5])
    color = (np.asarray(cmap(values)[:, 0:3]) * 255).astype(np.uint8)
    array_seq = get_data_as_arraysequence(color, fake_sft)

    colored_sft = add_data_as_color_dpp(
        fake_sft, array_seq)

    assert len(colored_sft.data_per_streamline.keys()) == 0
    assert list(colored_sft.data_per_point.keys()) == ['color']


def test_add_data_as_color_error_common_shape():

    fake_sft = _get_small_sft()

    # Test: One value per streamline
    # Should fail because the values aren't RGB values
    values = np.asarray([4, 5])
    array_seq = get_data_as_arraysequence(values, fake_sft)

    with pytest.raises(ValueError):
        _ = add_data_as_color_dpp(
            fake_sft, array_seq)


def test_add_data_as_color_error_number():

    fake_sft = _get_small_sft()
    cmap = get_lookup_table('jet')

    # Test: One value per streamline
    # Should fail because the values aren't RGB values
    values = np.asarray([2, 20, 200, 0.1, 0.3, 22, 5])
    array_seq = get_data_as_arraysequence(values, fake_sft)
    color = (np.asarray(cmap(values)[:, 0:3]) * 255).astype(np.uint8)
    color = color[:-2]  # Remove last streamline colors
    with pytest.raises(ValueError):
        _ = add_data_as_color_dpp(
            fake_sft, array_seq)


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

    # Average
    map_data = project_dpp_to_map(fake_sft, 'my_dpp')
    expected = np.zeros((3, 3, 3))  # fake_ref is 3x3x3
    expected[0, 0, 0] = 1  # the 3 points of the first streamline
    expected[1, 1, 1] = 2  # the 4 points of the second streamline
    assert np.array_equal(map_data, expected)

    # Sum
    map_data = project_dpp_to_map(fake_sft, 'my_dpp', sum_lines=True)
    expected = np.zeros((3, 3, 3))  # fake_ref is 3x3x3
    expected[0, 0, 0] = 3 * 1  # the 3 points of the first streamline
    expected[1, 1, 1] = 4 * 2  # the 4 points of the second streamline
    assert np.array_equal(map_data, expected)

    # Option 'endpoints_only':
    map_data = project_dpp_to_map(fake_sft, 'my_dpp', sum_lines=True,
                                  endpoints_only=True)
    expected = np.zeros((3, 3, 3))  # fake_ref is 3x3x3
    expected[0, 0, 0] = 2 * 1  # only 2 points of the first streamline
    expected[1, 1, 1] = 2 * 2  # only 2 points of the second streamline
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

    # Option 'endpoints only':
    dpp = perform_operation_on_dpp('max', fake_sft, 'my_dpp',
                                   endpoints_only=True)
    assert nan_array_equal(dpp[0].squeeze(), [1.0, np.nan, 1])
    assert nan_array_equal(dpp[1].squeeze(), [2.0, np.nan, 2])


def test_perform_operation_dpp_to_dps():
    fake_sft = _get_small_sft()

    # This fake dpp contains two values per point: [1, 0] at each point for the
    # first streamline (length 3), [2, 0] for the second (length 4).
    fake_sft.data_per_point['my_dpp'] = [[[1, 0]]*3,
                                         [[2, 0]]*4]

    # Operations are done separately for each value.

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

    # Option 'endpoints_only':
    dps = perform_operation_dpp_to_dps('sum', fake_sft, 'my_dpp',
                                       endpoints_only=True)
    assert np.array_equal(dps[0], [2 * 1, 0])
    assert np.array_equal(dps[1], [2 * 2, 0])


def test_perform_correlation_on_endpoints():
    fake_sft = _get_small_sft()
    fake_sft.data_per_point['my_dpp'] = [[[1, 0]]*3,
                                         [[2, 0], [0, 0], [0, 0], [0, 0]]]
    dps = perform_correlation_on_endpoints(fake_sft, 'my_dpp')
    assert np.allclose(dps[0], [1])
    assert nan_array_equal(dps[1], [np.nan])
