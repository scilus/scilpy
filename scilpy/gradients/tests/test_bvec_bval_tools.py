# -*- coding: utf-8 -*-
import numpy as np

from scilpy.gradients.bvec_bval_tools import round_bvals_to_shell


def test_is_normalized_bvecs():
    # toDO
    pass


def test_normalize_bvecs():
    # toDo
    pass


def test_check_b0_threshold():
    # toDo
    pass


def test_fsl2mrtrix():
    # toDo
    pass


def test_mrtrix2fsl():
    # toDo
    pass


def test_identify_shells():
    # toDo
    pass


def test_str_to_axis_index():
    # Very simple, nothing to do
    pass


def test_flip_gradient_sampling():
    # toDo
    pass


def test_swap_gradient_axis():
    # toDo
    pass


def test_round_bvals_to_shell():
    tolerance = 10

    # 1. Verify that works even with inverted shells
    bvals = np.asarray([0, 1, 1.5, 9, 1000, 991, 1009, 0, 0])
    shells = [1000, 0]
    out_bvals = round_bvals_to_shell(bvals, tolerance, shells)
    assert np.array_equal(out_bvals, [0, 0, 0, 0, 1000, 1000, 1000, 0, 0])

    # 2. Verify that doesn't work with value on the limit: data on inexpected
    # shell
    bvals = np.asarray([0, 10])
    shells = [0]
    success = True
    try:
        _ = round_bvals_to_shell(bvals, tolerance, shells)
    except ValueError:
        success = False
    assert not success

    # 3. Verify that doesn't work with shell missing: no data on shell 1000.
    bvals = np.asarray([0, 10])
    shells = [0, 1000]
    success = True
    try:
        _ = round_bvals_to_shell(bvals, tolerance, shells)
    except ValueError:
        success = False
    assert not success

    # 4. Verify that works with given shell not exactly on centroid
    bvals = np.asarray([5, 1, 1.5, 9, 1000, 991, 1009, 0, 0])
    shells = [0, 1000]
    out_bvals = round_bvals_to_shell(bvals, tolerance, shells)
    assert np.array_equal(out_bvals, [0, 0, 0, 0, 1000, 1000, 1000, 0, 0])

    # 5. KNOWN LIMIT: Very weird and rare case. The b-val 11 is accepted as 0,
    # because it is less than 10 from centroid 7, even though it is more than
    # 10 from expected shell 0. toDo
    bvals = np.asarray([7, 8, 9, 11, 0, 0])
    shells = [0]
    out_bvals = round_bvals_to_shell(bvals, tolerance, shells)
    assert np.array_equal(out_bvals, [0, 0, 0, 0, 0, 0])

    # 6. KNOWN BUG: [1000, 991, 1009] works. But [991, 1009, 1000] fails
    # because first centroid becomes 991, and then 1009 is out of tolerance.
    # toDo
    bvals = np.asarray([1000, 991, 1009])
    shells = [1000]
    out_bvals = round_bvals_to_shell(bvals, tolerance, shells)
    assert np.array_equal(out_bvals, [1000, 1000, 1000])
