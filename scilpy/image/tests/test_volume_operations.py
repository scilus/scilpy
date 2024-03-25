# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from numpy.testing import assert_equal, assert_almost_equal

from scilpy import SCILPY_HOME
from scilpy.image.volume_operations import (apply_transform,
                                            compute_snr,
                                            crop_volume,
                                            flip_volume,
                                            merge_metrics,
                                            normalize_metric,
                                            resample_volume)
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.utils.util import compute_nifti_bounding_box

# Fetching testing dwi data.
fetch_data(get_testing_files_dict(), keys='processing.zip')
tmp_dir = tempfile.TemporaryDirectory()
in_dwi = os.path.join(SCILPY_HOME, 'processing', 'dwi.nii.gz')
in_bval = os.path.join(SCILPY_HOME, 'processing', 'dwi.bval')
in_bvec = os.path.join(SCILPY_HOME, 'processing', 'dwi.bvec')
in_mask = os.path.join(SCILPY_HOME, 'processing', 'cc.nii.gz')
in_noise_mask = os.path.join(SCILPY_HOME, 'processing',
                             'small_roi_gm_mask.nii.gz')


def test_count_non_zero_voxel():
    # Not necessary since it is only using numpy.count_nonzero()

    pass


def test_flip_volume():
    vol = np.empty((3, 3, 3))
    vol[0, 0, 0] = 1

    vol_flipped_x = flip_volume(vol, 'x')
    vol_flipped_y = flip_volume(vol, 'y')
    vol_flipped_z = flip_volume(vol, 'z')

    assert vol[0, 0, 0] == vol_flipped_x[2, 0, 0]
    assert vol[0, 0, 0] == vol_flipped_y[0, 2, 0]
    assert vol[0, 0, 0] == vol_flipped_z[0, 0, 2]


def test_crop_volume():
    temp = np.ones((3, 3, 3))
    vol = np.pad(temp, pad_width=2, mode='constant', constant_values=0)

    img = nib.Nifti1Image(vol, np.eye(4))
    wbbox = compute_nifti_bounding_box(img)

    vol_cropped = crop_volume(img, wbbox)

    assert_equal(temp, vol_cropped.get_fdata())


def test_apply_transform():
    # Test apply transform on 3D volume.
    moving3d = np.pad(np.ones((3, 3, 3)), pad_width=1,
                      mode='constant', constant_values=0)
    ref3d = np.roll(moving3d, shift=(1, 0, 0), axis=(0, 1, 2))

    # Simulating translation by 1 in x-axis.
    transfo = np.eye(4)
    transfo[0, 3] = 1

    moving3d_img = nib.Nifti1Image(moving3d, np.eye(4))
    ref3d_img = nib.Nifti1Image(ref3d, np.eye(4))

    warped_img3d = apply_transform(transfo, ref3d_img, moving3d_img)

    assert_equal(ref3d, warped_img3d.get_fdata())

    # Test apply transform on 4D volume.
    moving4d = np.pad(np.ones((3, 3, 3, 2)), pad_width=1,
                      mode='constant', constant_values=0)

    moving4d_img = nib.Nifti1Image(moving4d, np.eye(4))

    warped_img4d = apply_transform(transfo, ref3d_img, moving4d_img)

    assert_equal(ref3d, warped_img4d.get_fdata()[:, :, :, 2])


def test_transform_dwi():
    # Tested within test_apply_transform().

    pass


def test_register_image():
    # Not necessary since it is mostly dipy's function.

    pass


def test_compute_snr():
    # Optimal unit test would be on perfect data with simulated noise but would
    # require making a dwi volume with known bvals and bvecs.
    dwi = nib.load(in_dwi)
    bvals, bvecs = read_bvals_bvecs(in_bval, in_bvec)
    mask = nib.load(in_mask)
    noise_mask = nib.load(in_noise_mask)

    snr = compute_snr(dwi, bvals, bvecs, 20, mask,
                      noise_mask=noise_mask, noise_map=None,
                      split_shells=True)

    # Value returned from multiple runs on the same data.
    target_val = 10.216334

    assert np.allclose(snr[0]['snr'], target_val, atol=0.00005)


def test_resample_volume():
    moving3d = np.pad(np.ones((4, 4, 4)), pad_width=1,
                      mode='constant', constant_values=0)
    ref3d = np.ones((2, 2, 2))

    moving3d_img = nib.Nifti1Image(moving3d, np.eye(4))

    resampled_img = resample_volume(moving3d_img, res=(2, 2, 2), interp='nn')

    assert_equal(resampled_img.get_fdata(), ref3d)


def test_normalize_metric_basic():
    metric = np.array([1, 2, 3, 4, 5])
    expected_output = np.array([0., 0.25, 0.5, 0.75, 1.])
    normalized_metric = normalize_metric(metric)
    assert_almost_equal(normalized_metric, expected_output)


def test_normalize_metric_nan_handling():
    metric = np.array([1, np.nan, 3, np.nan, 5])
    expected_output = np.array([0., np.nan, 0.5, np.nan, 1.])
    normalized_metric = normalize_metric(metric)

    assert_almost_equal(normalized_metric, expected_output)


def test_merge_metrics_basic():
    arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    # Geometric mean boosted by beta=1
    expected_output = np.array([2.0, 3.162278, 4.242641])
    merged_metric = merge_metrics(*arrays)

    assert_almost_equal(merged_metric, expected_output, decimal=6)


def test_merge_metrics_nan_propagation():
    arrays = [np.array([1, np.nan, 3]), np.array([4, 5, 6])]
    expected_output = np.array([2., np.nan, 4.242641])  # NaN replaced with -2
    merged_metric = merge_metrics(*arrays)

    assert_almost_equal(merged_metric, expected_output, decimal=6)
