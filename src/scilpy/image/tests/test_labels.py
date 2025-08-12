# -*- coding: utf-8 -*-

from copy import deepcopy
from glob import glob
import os

import nibabel as nib
import numpy as np
from numpy.testing import assert_equal
import pytest

from scilpy.image.labels import (combine_labels, dilate_labels,
                                 get_data_as_labels, get_labels_from_mask,
                                 remove_labels, split_labels)
from scilpy.tests.arrays import ref_in_labels, ref_out_labels


def test_combine_labels_all_labels():
    in_labels = deepcopy(ref_in_labels)
    out_labels = combine_labels([in_labels], [np.unique(ref_in_labels)],
                                ('all_labels',), background_id=0,
                                merge_groups=False)
    assert_equal(ref_in_labels, out_labels)


def test_combine_labels_out_labels_ids():
    in_labels = deepcopy(ref_in_labels)
    out_labels = combine_labels([in_labels], [[2, 4, 6]],
                                ('out_labels_ids', [1, 2, 3]), background_id=0,
                                merge_groups=False)

    exp_labels = deepcopy(ref_out_labels)
    exp_labels[exp_labels == 2] = 1
    exp_labels[exp_labels == 4] = 2
    exp_labels[exp_labels == 6] = 3

    assert_equal(out_labels, exp_labels)


def test_combine_labels_out_labels_ids_merge():
    in_labels = deepcopy(ref_in_labels)
    out_labels = combine_labels([in_labels], [[2, 4, 6]],
                                ('out_labels_ids', [2]), background_id=0,
                                merge_groups=True)

    exp_labels = deepcopy(ref_out_labels)
    exp_labels[exp_labels > 0] = 2

    assert_equal(out_labels, exp_labels)


def test_combine_labels_unique():
    in_labels = deepcopy(ref_in_labels)
    out_labels = combine_labels([in_labels], [[2, 4, 6]],
                                ('unique',), background_id=0,
                                merge_groups=False)

    exp_labels = deepcopy(ref_out_labels)
    exp_labels[exp_labels == 2] = 1
    exp_labels[exp_labels == 4] = 2
    exp_labels[exp_labels == 6] = 3

    assert_equal(out_labels, exp_labels)


def test_combine_labels_group_in_m():
    in_labels = deepcopy(ref_in_labels)
    out_labels = combine_labels([in_labels, in_labels], [[], [2, 4, 6]],
                                ('group_in_m',), background_id=0,
                                merge_groups=False)

    exp_labels = deepcopy(ref_out_labels)
    exp_labels[exp_labels == 2] = 2 + 1e4
    exp_labels[exp_labels == 4] = 4 + 1e4
    exp_labels[exp_labels == 6] = 6 + 1e4

    assert_equal(out_labels, exp_labels)


def test_dilate_labels_with_mask():
    in_labels = deepcopy(ref_in_labels)
    in_mask = deepcopy(ref_in_labels)
    in_mask[in_mask > 0] = 1
    out_labels = dilate_labels(in_labels, 1, 2, 1,
                               labels_to_dilate=[1, 6],
                               labels_not_to_dilate=[3, 4],
                               labels_to_fill=[0, 2, 5],
                               mask=in_mask)

    exp_labels = deepcopy(ref_in_labels)
    exp_labels[exp_labels == 2] = 1
    exp_labels[exp_labels == 5] = 6

    assert_equal(out_labels, exp_labels)


def test_dilate_labels_without_mask():
    in_labels = deepcopy(ref_in_labels)
    out_labels = dilate_labels(in_labels, 1, 2, 1,
                               labels_to_dilate=[1, 6],
                               labels_not_to_dilate=[3, 4, 5],
                               labels_to_fill=[0], mask=None)

    for i, val in enumerate([544, 156, 36, 36, 36, 36, 156]):
        assert len(out_labels[out_labels == i]) == val


def test_get_data_as_labels_int():
    data = np.zeros((2, 2, 2), dtype=np.int64)
    img = nib.Nifti1Image(data, np.eye(4), dtype=np.int64)
    img.set_filename('test.nii.gz')

    _ = get_data_as_labels(img)

    img.set_data_dtype(np.uint8)
    _ = get_data_as_labels(img)

    img.set_data_dtype(np.uint16)
    _ = get_data_as_labels(img)


def test_get_data_as_labels_float():
    data = np.zeros((2, 2, 2), dtype=np.float64)
    img = nib.Nifti1Image(data, np.eye(4))
    img.set_filename('test.nii.gz')

    with pytest.raises(Exception):
        _ = get_data_as_labels(img)

    img.set_data_dtype(np.float32)
    with pytest.raises(Exception):
        _ = get_data_as_labels(img)


def test_get_labels_from_mask():
    """ Test get_labels_from_mask with default labels. """
    # ref_out_labels contains disjoint blobs with values 2,4,6
    data = deepcopy(ref_out_labels)
    data[data == 2] = 1
    data[data == 4] = 2
    data[data == 6] = 3
    mask = data.astype(bool)

    labels = get_labels_from_mask(mask)

    assert_equal(labels, data)


def test_get_labels_from_mask_custom_labels_raises():
    """ Test get_labels_from_mask with custom labels. """
    # ref_out_labels contains disjoint blobs with values 2,4,6
    data = deepcopy(ref_out_labels)
    mask = data.astype(bool)
    labels = get_labels_from_mask(mask, [2, 4, 6, 8])

    assert np.unique(labels).size == 4  # including background


def test_get_labels_from_mask_custom_labels():
    """ Test get_labels_from_mask with custom labels. """
    # ref_out_labels contains disjoint blobs with values 2,4,6
    data = deepcopy(ref_out_labels)
    mask = data.astype(bool)

    labels = get_labels_from_mask(mask, [2, 4, 6])

    assert_equal(labels, data)


def test_get_labels_from_mask_custom_background():
    """ test get_labels_from_mask with custom background. """
    # ref_out_labels contains disjoint blobs with values 2,4,6
    data = deepcopy(ref_out_labels)
    mask = data.copy().astype(bool)

    data[data == 0] = 9

    labels = get_labels_from_mask(mask, [2, 4, 6], background_label=9)

    assert_equal(labels, data)


def test_remove_labels():
    in_labels = deepcopy(ref_in_labels)
    out_labels = remove_labels(in_labels, [3, 4, 5, 6, 7, 7])

    exp_labels = deepcopy(ref_in_labels)
    exp_labels[exp_labels >= 3] = 0

    assert_equal(out_labels, exp_labels)


def test_split_labels():
    in_labels = deepcopy(ref_in_labels)
    out_labels = split_labels(in_labels, [6, 7, 7])

    assert len(out_labels) == 3
    assert_equal(np.unique(out_labels[0]), [0, 6])
    assert_equal(np.unique(out_labels[1]), [0])


def test_stats_in_labels():
    # toDO. Will need to create a fake LUT.
    pass
