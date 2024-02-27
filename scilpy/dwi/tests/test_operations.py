# -*- coding: utf-8 -*-
import numpy as np

from scilpy.dwi.operations import compute_dwi_attenuation, \
    detect_volume_outliers, apply_bias_field


def test_apply_bias_field():

    # DWI is 1 everywhere, one voxel at 0.
    dwi = np.ones((10, 10, 10, 5))
    dwi[0, 0, 0, :] = 0
    mask = np.ones((10, 10, 10), dtype=bool)

    # bias field is 2 everywhere
    bias_field = 2 * np.ones((10, 10, 10))

    # result should be 1/2 everywhere, one voxel at 0. Rescaled to 0-1.
    out_dwi = apply_bias_field(dwi, bias_field, mask)
    assert np.max(out_dwi) == 1
    assert out_dwi[0, 0, 0, 0] == 0


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
    # For this test: all 90 or 180 degrees on one shell.
    bvals = 1000 * np.ones(5)
    bvecs = np.asarray([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0],  # inverse of first
                        [0, -1, 0]])  # inverse of second

    # DWI associated with the last bvec is very different. Others are highly
    # correlated (but not equal, or the correlation is NaN: one voxel
    # different). One voxel different for the first 4 gradients. Random for
    # the last.
    dwi = np.ones((10, 10, 10, 5))
    dwi[0, 0, 0, 0:4] = np.random.rand(4)
    dwi[..., -1] = np.random.rand(10, 10, 10)

    res, outliers = detect_volume_outliers(dwi, bvals, bvecs, std_scale=1)

    # Should get one shell
    keys = list(res.keys())
    assert len(keys) == 1
    assert keys[0] == 1000
    res = res[1000]
    outliers = outliers[1000]

    # Should get a table 5x3.
    assert np.array_equal(res.shape, [5, 3])

    # First column: index of the bvecs. They should all be managed.
    assert np.array_equal(np.sort(res[:, 0]), np.arange(5))

    # Second column = Mean angle. The most different should be the 3rd (#2)
    # But not an outlier.
    assert np.argmax(res[:, 1]) == 2
    assert len(outliers['outliers_angle']) == 0

    # Thirst column = corr. The most uncorrelated should be the 5th (#4)
    # Should also be an outlier with STD 1
    assert np.argmin(res[:, 2]) == 4
    assert outliers['outliers_corr'][0] == 4


def test_compute_residuals():
    # Quite simple. Not testing.
    pass


def test_compute_residuals_statistics():
    # Simply calling numpy methods. Nothing to test.
    pass
