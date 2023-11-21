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
    out_bvals = round_bvals_to_shell(bvals, shells, tol=tolerance)
    assert np.array_equal(out_bvals, [0, 0, 0, 0, 1000, 1000, 1000, 0, 0])

    # 2. Verify that doesn't work with value on the limit: data on inexpected
    # shell
    bvals = np.asarray([0, 11])
    shells = [0]
    success = True
    try:
        _ = round_bvals_to_shell(bvals, shells, tol=tolerance)
    except ValueError:
        success = False
    assert not success

    # 3. Verify that doesn't work with shell missing: no data on shell 1000.
    bvals = np.asarray([0, 10])
    shells = [0, 1000]
    success = True
    try:
        _ = round_bvals_to_shell(bvals, shells, tol=tolerance)
    except ValueError:
        success = False
    assert not success
