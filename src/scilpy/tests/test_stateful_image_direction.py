# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
from scilpy.io.stateful_image import StatefulImage
from scilpy.reconst.utils import is_data_peaks

def test_peak_direction_transform():
    # Create a 90-degree rotation affine (X-axis)
    # y_world = -z_voxel, z_world = y_voxel
    affine = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    # 1. Test Peaks (3 coefficients)
    data_peaks = np.zeros((2, 2, 2, 3))
    data_peaks[:, :, :, :] = [0, 0, 1] # Voxel Z
    
    img = nib.Nifti1Image(data_peaks, affine)
    simg = StatefulImage.convert_to_simg(img)
    
    # Voxel (0,0,1) -> World (0,-1,0)
    world_peaks = simg.to_world_direction(data_peaks)
    expected_world = [0, -1, 0]
    np.testing.assert_allclose(world_peaks[0, 0, 0], expected_world, atol=1e-5)
    
    # World (0,-1,0) -> Voxel (0,0,1)
    voxel_peaks = simg.to_voxel_direction(world_peaks)
    expected_voxel = [0, 0, 1]
    np.testing.assert_allclose(voxel_peaks[0, 0, 0], expected_voxel, atol=1e-5)


def test_sh_direction_transform():
    # Create a 90-degree rotation affine (X-axis)
    affine = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    # Order 2, 6 coefficients for symmetric
    data_sh = np.zeros((2, 2, 2, 6))
    data_sh[:, :, :, 0] = 5.0 # Isotropic part, make sure it's the max so it's recognized as SH
    data_sh[:, :, :, 1:] = 0.01 # Add noise to prevent exact zeros
    data_sh[:, :, :, 3] = 1.0 # Some orientation part

    img = nib.Nifti1Image(data_sh, affine)
    simg = StatefulImage.convert_to_simg(img)

    # Verify it doesn't crash and changes coefficients
    world_sh = simg.to_world_direction(data_sh)
    assert not np.allclose(world_sh[0, 0, 0], data_sh[0, 0, 0])

    # Reverting should return original
    back_sh = simg.to_voxel_direction(world_sh)
    np.testing.assert_allclose(back_sh, data_sh, atol=1e-5)


def test_stateful_image_load_direction(tmp_path):
    affine = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    data_peaks = np.zeros((2, 2, 2, 3))
    data_peaks[:, :, :, :] = [0, 0, 1] # Voxel Z

    img_path = str(tmp_path / "voxel_peaks.nii.gz")
    nib.save(nib.Nifti1Image(data_peaks, affine), img_path)

    # Load as voxel-space directional image
    # Internal representation should move to World Space (0, -1, 0)
    simg = StatefulImage.load(img_path, is_orientation=True, is_world_space=False)

    expected_world = [0, -1, 0]
    np.testing.assert_allclose(simg.get_fdata()[0, 0, 0], expected_world, atol=1e-5)


def test_heuristic_is_data_peaks():
    # Peaks: multiple peaks with zeros or high argmax
    peaks_data = np.zeros((2, 2, 2, 6))
    # Make sure the max is in the first triplet to pass the `argmax_indices > 2` check
    # But place it at index 1 to trigger the `== 1 or == 2` check
    peaks_data[0, 0, 0, :3] = [0, 1, 0] # Peak 1 is Y
    # Argmax is 1 -> is_peaks should be True
    assert is_data_peaks(peaks_data) is True

    # SH: First value (l=0) is usually highest
    sh_data = np.zeros((2, 2, 2, 6))
    sh_data[:, :, :, 0] = 1.0 # l=0
    sh_data[:, :, :, 1:] = 0.1 # Small l=2
    # Argmax is 0 -> is_peaks should be False
    assert is_data_peaks(sh_data) is False
