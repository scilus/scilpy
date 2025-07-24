# -*- coding: utf-8 -*-
import numpy as np

from scilpy.gradients.bvec_bval_tools import (
    check_b0_threshold, identify_shells, is_normalized_bvecs,
    flip_gradient_sampling, normalize_bvecs, round_bvals_to_shell,
    str_to_axis_index, swap_gradient_axis)

bvecs = np.asarray([[1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [8.0, 1.0, 1.0]])


def test_is_normalized_bvecs():
    assert not is_normalized_bvecs(bvecs)
    assert is_normalized_bvecs(
        bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True))


def test_normalize_bvecs():
    assert is_normalized_bvecs(normalize_bvecs(bvecs))


def test_check_b0_threshold():
    assert check_b0_threshold(min_bval=0, b0_thr=0, skip_b0_check=False) == 0
    assert check_b0_threshold(min_bval=0, b0_thr=20, skip_b0_check=False) == 20
    assert check_b0_threshold(min_bval=20, b0_thr=0, skip_b0_check=True) == 20

    error_raised = False
    try:
        _ = check_b0_threshold(min_bval=20, b0_thr=0, skip_b0_check=False)
    except ValueError:
        error_raised = True
    assert error_raised


def test_identify_shells():
    def _subtest_identify_shells(bvals, threshold,
                                 expected_raw_centroids, expected_raw_shells,
                                 expected_round_sorted_centroids,
                                 expected_round_sorted_shells):
        bvals = np.asarray(bvals)

        # 1) Not rounded, not sorted
        c, s = identify_shells(bvals, threshold)
        assert np.array_equal(c, expected_raw_centroids)
        assert np.array_equal(s, expected_raw_shells)

        # 2) Rounded, sorted
        c, s = identify_shells(bvals, threshold, round_centroids=True,
                               sort=True)
        assert np.array_equal(c, expected_round_sorted_centroids)
        assert np.array_equal(s, expected_round_sorted_shells)

    # Test 1. All easy. Over the limit for 0, 5, 15. Clear difference for
    # 100, 2000.
    _subtest_identify_shells(bvals=[0, 0, 5, 15, 2000, 100], threshold=50,
                             expected_raw_centroids=[0, 2000, 100],
                             expected_raw_shells=[0, 0, 0, 0, 1, 2],
                             expected_round_sorted_centroids=[0, 100, 2000],
                             expected_round_sorted_shells=[0, 0, 0, 0, 2, 1])

    # Test 2. Threshold on the limit.
    # Additional difficulty with option rounded: two shells with the same
    # value, but a warning is printed. Should it raise an error?
    _subtest_identify_shells(bvals=[0, 0, 5, 2000, 100], threshold=5,
                             expected_raw_centroids=[0, 5, 2000, 100],
                             expected_raw_shells=[0, 0, 1, 2, 3],
                             expected_round_sorted_centroids=[0, 0, 100, 2000],
                             expected_round_sorted_shells=[0, 0, 1, 3, 2])


def test_str_to_axis_index():
    # Very simple, nothing to do
    assert str_to_axis_index('x') == 0
    assert str_to_axis_index('y') == 1
    assert str_to_axis_index('z') == 2
    assert str_to_axis_index('v') is None


def test_flip_gradient_sampling():
    fsl_bvecs = bvecs.T
    b = flip_gradient_sampling(fsl_bvecs, axes=[0], sampling_type='fsl')
    assert np.array_equal(b, np.asarray([[-1.0, 1.0, 1.0],
                                         [-1.0, 0.0, 1.0],
                                         [-0.0, 1.0, 0.0],
                                         [-8.0, 1.0, 1.0]]).T)


def test_swap_gradient_axis():
    fsl_bvecs = bvecs.T
    final_order = [1, 0, 2]
    b = swap_gradient_axis(fsl_bvecs, final_order, sampling_type='fsl')
    assert np.array_equal(b, np.asarray([[1.0, 1.0, 1.0],
                                         [0.0, 1.0, 1.0],
                                         [1.0, 0.0, 0.0],
                                         [1.0, 8.0, 1.0]]).T)


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
