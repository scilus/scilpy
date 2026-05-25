# -*- coding: utf-8 -*-
import numpy as np
import pytest

from scilpy.reconst.utils import (compute_max_sf_amplitude,
                                  compute_sf_threshold_mask)
from scilpy.tests.arrays import fodf_3x3_order8_descoteaux07


def test_compute_max_sf_amplitude():
    # Test with SH data
    max_sf = compute_max_sf_amplitude(fodf_3x3_order8_descoteaux07,
                                      sh_basis='descoteaux07',
                                      is_legacy=True)
    assert max_sf.shape == (3, 3, 1)
    assert np.all(max_sf >= 0)

    # Test with mask
    mask = np.zeros((3, 3, 1), dtype=bool)
    mask[1, 1, 0] = True
    max_sf_masked = compute_max_sf_amplitude(fodf_3x3_order8_descoteaux07,
                                             sh_basis='descoteaux07',
                                             is_legacy=True,
                                             mask=mask)
    assert np.count_nonzero(max_sf_masked) == 1
    assert max_sf_masked[1, 1, 0] == max_sf[1, 1, 0]


def test_compute_sf_threshold_mask_sh():
    # Test relative threshold
    mask, global_max, threshold = compute_sf_threshold_mask(
        fodf_3x3_order8_descoteaux07, relative_factor=0.5,
        sh_basis='descoteaux07', is_legacy=True, postprocess_mask=False)

    assert mask.shape == (3, 3, 1)
    assert global_max == np.max(compute_max_sf_amplitude(
        fodf_3x3_order8_descoteaux07, sh_basis='descoteaux07', is_legacy=True))
    assert threshold == 0.5 * global_max
    assert np.all(mask == (compute_max_sf_amplitude(
        fodf_3x3_order8_descoteaux07, sh_basis='descoteaux07',
        is_legacy=True) >= threshold))

    # Test absolute threshold
    mask, global_max, threshold = compute_sf_threshold_mask(
        fodf_3x3_order8_descoteaux07, absolute_threshold=0.1,
        sh_basis='descoteaux07', is_legacy=True, postprocess_mask=False)
    assert threshold == 0.1


def test_compute_sf_threshold_mask_peaks():
    # Test with 4D peaks (3*npeaks)
    peaks_4d = np.zeros((3, 3, 1, 6))  # 2 peaks
    peaks_4d[1, 1, 0, :3] = [1, 0, 0]  # norm 1
    peaks_4d[2, 2, 0, 3:] = [0, 0, 2]  # norm 2

    mask, global_max, threshold = compute_sf_threshold_mask(
        peaks_4d, relative_factor=0.6, postprocess_mask=False)

    assert global_max == 2.0
    assert threshold == 1.2
    assert np.count_nonzero(mask) == 1
    assert mask[2, 2, 0]

    # Test with 5D peaks (npeaks, 3)
    peaks_5d = np.zeros((3, 3, 1, 2, 3))
    peaks_5d[1, 1, 0, 0, :] = [1, 0, 0]
    peaks_5d[2, 2, 0, 1, :] = [0, 0, 2]

    mask, global_max, threshold = compute_sf_threshold_mask(
        peaks_5d, relative_factor=0.6, postprocess_mask=False)
    assert global_max == 2.0
    assert np.count_nonzero(mask) == 1


def test_compute_sf_threshold_mask_edge_cases():
    # Test relative_factor validation
    with pytest.raises(ValueError):
        compute_sf_threshold_mask(fodf_3x3_order8_descoteaux07,
                                  relative_factor=1.5)

    # Test zero data
    zero_data = np.zeros((3, 3, 1, 45))
    mask, global_max, threshold = compute_sf_threshold_mask(
        zero_data, relative_factor=0.5, sh_basis='descoteaux07',
        is_legacy=True)
    assert global_max == 0
    assert not np.any(mask)

    # Test no params
    with pytest.raises(ValueError):
        compute_sf_threshold_mask(fodf_3x3_order8_descoteaux07)


def test_compute_sf_threshold_mask_postprocess():
    # Create a mask with two components
    data = np.zeros((10, 10, 10, 6))  # 4D peaks
    data[2:5, 2:5, 2:5, :3] = [1, 0, 0]  # Large component
    data[8, 8, 8, :3] = [1, 0, 0]  # Small component

    mask, _, _ = compute_sf_threshold_mask(
        data, relative_factor=0.5, postprocess_mask=True)

    # Only large component should remain
    assert np.count_nonzero(mask) == 27
    assert not mask[8, 8, 8]
    assert mask[3, 3, 3]
