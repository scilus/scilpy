# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
from numpy.testing import (assert_,
                           assert_equal,
                           assert_array_almost_equal,
                           assert_raises)
import pytest

from scilpy.image.labels import (combine_labels, dilate_labels,
                                 get_data_as_labels, get_lut_dir,
                                 remove_labels, split_labels)

ref_in_labels = np.zeros((10, 10, 10), dtype=np.uint16)
for i in range(2, 8):
    ref_in_labels[2:8, 2:8, i] = i

ref_out_labels = deepcopy(ref_in_labels)
for i in range(1, 8, 2):
    ref_out_labels[ref_out_labels == i] = 0

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
