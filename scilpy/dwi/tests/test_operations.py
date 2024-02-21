# -*- coding: utf-8 -*-
import numpy as np

from scilpy.dwi.operations import compute_dwi_attenuation


def test_apply_bias_field():
    pass


def test_compute_dwi_attenuation():
    fake_b0 = np.ones((10, 10, 10))
    fake_dwi = np.ones((10, 10, 10, 4)) * 0.5

    # Test 1: attenuation of 0.5 / 1 = 0.5 everywhere
    res = compute_dwi_attenuation(fake_dwi, fake_b0)
    expected = np.ones((10, 10, 10, 4)) * 0.5
    assert np.array_equal(res, expected)

    # Test 2: noisy data: one voxel is not attenuated, and has a value > b0 for
    # one gradient. Should give attenuation=1.
    fake_dwi[2, 2, 2, 2] = 2
    expected[2, 2, 2, 2] = 1

    # + Test 3: a 0 in the b0. Can divide correctly?
    fake_b0[4, 4, 4] = 0
    expected[4, 4, 4, :] = 0

    res = compute_dwi_attenuation(fake_dwi, fake_b0)
    assert np.array_equal(res, expected)


def test_detect_volume_outliers():
    pass


def test_compute_residuals():
    pass


def test_compute_residuals_statistics():
    # Simply calling numpy methods. Nothing to test.
    pass
