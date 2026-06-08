# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
from scilpy.io.stateful_image import StatefulImage


def test_stateful_image_save_world_vs_voxel(tmp_path):
    # Create a 90-degree rotation affine (X-axis)
    # y_world = -z_voxel, z_world = y_voxel
    affine = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    # Peaks (3 coefficients)
    # Data is Z in voxel space -> (0, -1, 0) in world space
    data_peaks = np.zeros((2, 2, 2, 3))
    data_peaks[:, :, :, :] = [0, 0, 1]

    img_path = str(tmp_path / "original.nii.gz")
    nib.save(nib.Nifti1Image(data_peaks, affine), img_path)

    # Load. Internal data becomes world space: [0, -1, 0]
    simg = StatefulImage.load(img_path, is_orientation=True,
                              is_world_space=False)

    # 1. Save in World Space (Default)
    world_save_path = str(tmp_path / "world_save.nii.gz")
    simg.save(world_save_path, in_world_space=True)

    # Load back with MRtrix assumption (is_world_space=True)
    simg_world = StatefulImage.load(world_save_path, is_orientation=True,
                                    is_world_space=True)
    # Should still be [0, -1, 0] in world space
    np.testing.assert_allclose(simg_world.get_fdata()[0, 0, 0], [0, -1, 0],
                               atol=1e-5)
    # The raw data in the NIfTI file should be [0, -1, 0]
    raw_world_data = nib.load(world_save_path).get_fdata()
    np.testing.assert_allclose(raw_world_data[0, 0, 0], [0, -1, 0], atol=1e-5)

    # 2. Save in Voxel Space (Old Dipy style)
    voxel_save_path = str(tmp_path / "voxel_save.nii.gz")
    simg.save(voxel_save_path, in_world_space=False)

    # Load back with Dipy assumption (is_world_space=False)
    simg_voxel = StatefulImage.load(voxel_save_path, is_orientation=True,
                                    is_world_space=False)
    # Should still be [0, -1, 0] in world space
    np.testing.assert_allclose(simg_voxel.get_fdata()[0, 0, 0], [0, -1, 0],
                               atol=1e-5)
    # raw_voxel_data[0, 0, 0] should be [0, 0, 1] (back to voxel Z)
    raw_voxel_data = nib.load(voxel_save_path).get_fdata()
    np.testing.assert_allclose(raw_voxel_data[0, 0, 0], [0, 0, 1], atol=1e-5)


def test_stateful_image_save_reoriented(tmp_path):
    # Test saving when the in-memory image is reoriented
    original_affine = np.eye(4)  # RAS
    data_peaks = np.zeros((2, 2, 2, 3))
    data_peaks[:, :, :, :] = [1, 0, 0]  # X (Right)

    img_path = str(tmp_path / "ras.nii.gz")
    nib.save(nib.Nifti1Image(data_peaks, original_affine), img_path)

    # Load and reorient to LAS
    simg = StatefulImage.load(img_path, to_orientation="LAS",
                              is_orientation=True, is_world_space=False)

    # In LAS, Right is -X. So in-memory data (voxel space) should be [-1, 0, 0]
    # Wait, load(is_world_space=False) rotates to world space on load.
    # World space [1, 0, 0] is always Right.
    # So simg.get_fdata() should be [1, 0, 0] (World Space)
    np.testing.assert_allclose(simg.get_fdata()[0, 0, 0], [1, 0, 0], atol=1e-5)

    # Save back to original (RAS).
    # 1. World Space save
    save_world = str(tmp_path / "save_world.nii.gz")
    simg.save(save_world, in_world_space=True)
    raw_world = nib.load(save_world).get_fdata()
    # Should be [1, 0, 0]
    np.testing.assert_allclose(raw_world[0, 0, 0], [1, 0, 0], atol=1e-5)

    # 2. Voxel Space save
    save_voxel = str(tmp_path / "save_voxel.nii.gz")
    simg.save(save_voxel, in_world_space=False)
    raw_voxel = nib.load(save_voxel).get_fdata()
    # Original orientation was RAS, so voxel X is Right.
    # Should be [1, 0, 0]
    np.testing.assert_allclose(raw_voxel[0, 0, 0], [1, 0, 0], atol=1e-5)

    # Let's try if original orientation was LAS
    las_affine = np.diag([-1, 1, 1, 1])
    las_path = str(tmp_path / "las.nii.gz")
    # Voxel [1, 0, 0] in LAS means Left.
    nib.save(nib.Nifti1Image(data_peaks, las_affine), las_path)

    simg_las = StatefulImage.load(las_path, to_orientation="RAS",
                                  is_orientation=True, is_world_space=False)
    # Load(is_world_space=False) -> rotates voxel [1, 0, 0] to world.
    # In LAS, voxel [1, 0, 0] is world [-1, 0, 0] (Left).
    np.testing.assert_allclose(simg_las.get_fdata()[0, 0, 0], [-1, 0, 0],
                               atol=1e-5)

    # Save back to original (LAS)
    # Voxel space save: should be [1, 0, 0] (Voxel space of LAS)
    simg_las.save(las_path, in_world_space=False)
    raw_las_voxel = nib.load(las_path).get_fdata()
    np.testing.assert_allclose(raw_las_voxel[0, 0, 0], [1, 0, 0], atol=1e-5)

    # World space save: should be [-1, 0, 0] (World space)
    simg_las.save(las_path, in_world_space=True)
    raw_las_world = nib.load(las_path).get_fdata()
    np.testing.assert_allclose(raw_las_world[0, 0, 0], [-1, 0, 0], atol=1e-5)
