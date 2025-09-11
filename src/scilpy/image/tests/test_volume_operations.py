# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from numpy.testing import assert_equal, assert_almost_equal

from scilpy import SCILPY_HOME
from scilpy.image.volume_operations import (apply_transform,
                                            compute_distance_map, compute_snr,
                                            crop_volume, flip_volume,
                                            mask_data_with_default_cube,
                                            compute_nawm,
                                            merge_metrics, normalize_metric,
                                            resample_volume, reshape_volume,
                                            register_image)
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.image.utils import compute_nifti_bounding_box

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
    # Not necessary since it is mostly dipy's function, but running for the
    # sake of coverage, and because there are many options to give.
    static = np.ones((8, 8, 8))
    moving = np.ones((8, 8, 8))

    # Images are similar: one column of 'color', but not the same.
    # (We want to ensure convergence).
    static[1, :, :] = 2
    moving[2, :, :] = 2

    # Realistic affines, already close except in x
    static_grid2world = np.eye(4, 4, dtype=float)
    static_grid2world[:, 3] = [-22, -20, 10, 1]
    moving_grid2world = np.eye(4, 4, dtype=float)
    static_grid2world[:, 3] = [-20, -20, 10, 1]

    transformation_type = 'affine'
    dwi = np.ones((8, 8, 8, 2))
    dwi[2, :, :, :] = 2
    fine = True
    out_im, transform = register_image(static, static_grid2world, moving,
                                       moving_grid2world, transformation_type,
                                       dwi, fine)

    assert np.array_equal(transform.shape, [4, 4])
    assert np.array_equal(out_im.shape, [8, 8, 8, 2])

    # Input image had 2 values: 1, 2. With interpolation, we now have ~1, ~2
    # and the values on edges: a bit <1 and a bit>1.
    assert len(np.unique(np.round(out_im, decimals=4))) == 4


def test_compute_snr():
    # Optimal unit test would be on perfect data with simulated noise but would
    # require making a dwi volume with known bvals and bvecs.
    dwi = nib.load(in_dwi)
    bvals, bvecs = read_bvals_bvecs(in_bval, in_bvec)
    mask = nib.load(in_mask)
    noise_mask = nib.load(in_noise_mask)

    snr, _ = compute_snr(dwi, bvals, bvecs, 20, mask,
                         noise_mask=noise_mask, noise_map=None,
                         split_shells=True)

    # Value returned from multiple runs on the same data.
    target_val = 10.216334
    assert np.allclose(snr[0]['snr'], target_val, atol=0.00005)

    # Testing automatic noise mask (when giving no noise_mask, noise_map)
    # Current chosen dwi has no noise in the background. Adding manually. Let's
    # change the input someday.
    # Adding at the top (in z)
    dwi_data = dwi.get_fdata()
    dwi_data[:, :, -1, :] = np.random.rand(dwi_data.shape[0],
                                           dwi_data.shape[1],
                                           dwi_data.shape[3])
    dwi = nib.Nifti1Image(dwi_data, dwi.affine)
    snr, noise_mask = compute_snr(dwi, bvals, bvecs, 20, mask,
                                  noise_mask=None, noise_map=None,
                                  split_shells=True)

    # Value returned from multiple runs on the same data varies because of
    # random value. But always high (~ 11 600)
    assert snr[0]['snr'] > 5000


def test_remove_outliers_ransac():
    # Could test, but uses mainly sklearn. Not testing again.
    pass


def smooth_to_fwhm():
    # toDo
    pass


def test_resample_volume():
    # Input image: 6x6x6 (4x4x4 with zeros around)
    # affine as np.eye => voxel size 1x1x1
    moving3d = np.pad(np.ones((4, 4, 4)), pad_width=1,
                      mode='constant', constant_values=0)
    moving3d_img = nib.Nifti1Image(moving3d, np.eye(4))

    # Ref: 2x2x2, voxel size 3x3x3
    ref3d = np.ones((2, 2, 2))
    ref_affine = np.eye(4)*3
    ref_affine[-1, -1] = 1

    # 1) Option volume_shape: expecting an output of 2x2x2, which means
    # voxel resolution 3x3x3
    resampled_img = resample_volume(moving3d_img, volume_shape=(2, 2, 2),
                                    interp='nn')
    assert_equal(resampled_img.get_fdata(), ref3d)
    assert resampled_img.affine[0, 0] == 3

    # 2) Option reference image that is 2x2x2, resolution 3x3x3.
    ref_img = nib.Nifti1Image(ref3d, ref_affine)
    resampled_img = resample_volume(moving3d_img, ref_img=ref_img,
                                    interp='nn')
    assert_equal(resampled_img.get_fdata(), ref3d)
    assert resampled_img.affine[0, 0] == 3

    # 3) Option final resolution 3x3x3, should be of shape 2x2x2
    resampled_img = resample_volume(moving3d_img, voxel_res=(3, 3, 3),
                                    interp='nn')
    assert_equal(resampled_img.get_fdata(), ref3d)
    assert resampled_img.affine[0, 0] == 3

    # 4) Same test, with a fake 4th dimension
    moving3d = np.stack((moving3d, moving3d), axis=-1)
    moving3d_img = nib.Nifti1Image(moving3d, np.eye(4))
    resampled_img = resample_volume(moving3d_img, voxel_res=(3, 3, 3),
                                    interp='nn')
    result = resampled_img.get_fdata()
    assert_equal(result[:, :, :, 0], ref3d)
    assert_equal(result[:, :, :, 1], ref3d)
    assert resampled_img.affine[0, 0] == 3


def test_reshape_volume_pad():
    # 3D img
    img = nib.Nifti1Image(
        np.arange(1, (3**3)+1).reshape((3, 3, 3)).astype(float),
        np.eye(4))

    # 1) Reshaping to 4x4x4, padding with 0
    reshaped_img = reshape_volume(img, (4, 4, 4))

    assert_equal(reshaped_img.affine[:, -1], [-1, -1, -1, 1])
    assert_equal(reshaped_img.get_fdata()[0, 0, 0], 0)

    # 2) Reshaping to 4x4x4, padding with -1
    reshaped_img = reshape_volume(img, (4, 4, 4), mode='constant',
                                  cval=-1)
    assert_equal(reshaped_img.get_fdata()[0, 0, 0], -1)

    # 3) Reshaping to 4x4x4, padding with edge
    reshaped_img = reshape_volume(img, (4, 4, 4), mode='edge')
    assert_equal(reshaped_img.get_fdata()[0, 0, 0], 1)

    # 4D img (2 "stacked" 3D volumes)
    img = nib.Nifti1Image(
        np.arange(1, ((3**3) * 2)+1).reshape((3, 3, 3, 2)).astype(float),
        np.eye(4))

    # 2) Reshaping to 5x5x5, padding with 0
    reshaped_img = reshape_volume(img, (5, 5, 5))
    assert_equal(reshaped_img.get_fdata()[0, 0, 0, 0], 0)


def test_reshape_volume_crop():
    # 3D img
    img = nib.Nifti1Image(
        np.arange(1, (3**3)+1).reshape((3, 3, 3)).astype(float),
        np.eye(4))

    # 1) Cropping to 1x1x1
    reshaped_img = reshape_volume(img, (1, 1, 1))
    assert_equal(reshaped_img.get_fdata().shape, (1, 1, 1))
    assert_equal(reshaped_img.affine[:, -1], [1, 1, 1, 1])
    assert_equal(reshaped_img.get_fdata()[0, 0, 0], 14)

    # 2) Cropping to 2x2x2
    reshaped_img = reshape_volume(img, (2, 2, 2))
    assert_equal(reshaped_img.get_fdata().shape, (2, 2, 2))
    assert_equal(reshaped_img.affine[:, -1], [0, 0, 0, 1])
    assert_equal(reshaped_img.get_fdata()[0, 0, 0], 1)

    # 4D img
    img = nib.Nifti1Image(
        np.arange(1, ((3**3) * 2)+1).reshape((3, 3, 3, 2)).astype(float),
        np.eye(4))

    # 2) Cropping to 2x2x2
    reshaped_img = reshape_volume(img, (2, 2, 2))
    assert_equal(reshaped_img.get_fdata().shape, (2, 2, 2, 2))
    assert_equal(reshaped_img.affine[:, -1], [0, 0, 0, 1])
    assert_equal(reshaped_img.get_fdata()[0, 0, 0, 0], 1)


def test_reshape_volume_dtype():
    # 3D img
    img = nib.Nifti1Image(
        np.arange(1, (3**3)+1).reshape((3, 3, 3)).astype(np.uint16),
        np.eye(4))

    # 1) Staying in 3x3x3, same dtype
    reshaped_img = reshape_volume(img, (3, 3, 3))
    assert_equal(reshaped_img.get_fdata().shape, (3, 3, 3))
    assert reshaped_img.get_data_dtype() == img.get_data_dtype()

    # 1) Staying in 3x3x3, casting to float
    reshaped_img = reshape_volume(img, (3, 3, 3), dtype=float)
    assert_equal(reshaped_img.get_fdata().shape, (3, 3, 3))
    assert reshaped_img.get_data_dtype() == float


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


def test_mask_data_with_default_cube():
    data = np.ones((12, 12, 12))
    out = mask_data_with_default_cube(data)
    assert np.array_equal(data.shape, out.shape)
    assert out[0, 0, 0] == 0
    assert out[-1, -1, -1] == 0
    assert out[6, 6, 6] == 1


def test_distance_map_smallest_first():
    mask_1 = np.zeros((3, 3, 3))
    mask_1[0, 0, 0] = 1

    mask_2 = np.zeros((3, 3, 3))
    mask_2[1:3, 1:3, 1:3] = 1

    distance = compute_distance_map(mask_1, mask_2)
    assert np.abs(np.sum(distance) - 1.732050) < 1e-6


def test_compute_distance_map_biggest_first():
    # Swap both masks
    mask_2 = np.zeros((3, 3, 3))
    mask_2[0, 0, 0] = 1

    mask_1 = np.zeros((3, 3, 3))
    mask_1[1:3, 1:3, 1:3] = 1

    distance = compute_distance_map(mask_1, mask_2)
    assert np.abs(np.sum(distance) - 21.544621) < 1e-6


def test_compute_distance_map_symmetric():
    mask_1 = np.zeros((3, 3, 3))
    mask_1[0, 0, 0] = 1

    mask_2 = np.zeros((3, 3, 3))
    mask_2[1:3, 1:3, 1:3] = 1

    distance = compute_distance_map(mask_1, mask_2, symmetric=True)
    assert np.abs(np.sum(distance) - 23.276672) < 1e-6


def test_compute_distance_map_overlap():
    mask_1 = np.zeros((3, 3, 3))
    mask_1[1, 1, 1] = 1

    mask_2 = np.zeros((3, 3, 3))
    mask_2[1:3, 1:3, 1:3] = 1

    distance = compute_distance_map(mask_1, mask_2)
    assert np.all(distance == 0)


def test_compute_distance_map_wrong_shape():
    mask_1 = np.zeros((3, 3, 3))
    mask_2 = np.zeros((3, 3, 4))

    # Different shapes, test should fail
    try:
        compute_distance_map(mask_1, mask_2)
        assert False
    except ValueError:
        assert True


def test_compute_nawm_3D():
    lesion_img = np.zeros((3, 3, 3))
    lesion_img[1, 1, 1] = 1

    nawm = compute_nawm(lesion_img, nb_ring=0, ring_thickness=2)
    assert np.sum(nawm) == 1

    try:
        nawm = compute_nawm(lesion_img, nb_ring=2, ring_thickness=0)
        assert False
    except ValueError:
        assert True

    nawm = compute_nawm(lesion_img, nb_ring=1, ring_thickness=2)
    assert np.sum(nawm) == 53


def test_compute_nawm_4D():
    lesion_img = np.zeros((10, 10, 10))
    lesion_img[4, 4, 4] = 1
    lesion_img[2, 2, 2] = 2

    nawm = compute_nawm(lesion_img, nb_ring=2, ring_thickness=1)
    assert nawm.shape == (10, 10, 10, 2)
    val, count = np.unique(nawm[..., 0], return_counts=True)
    assert np.array_equal(val, [0, 1, 2, 3])
    assert np.array_equal(count, [967, 1, 6, 26])
