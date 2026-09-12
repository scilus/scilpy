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

    # Load. Internal data stays as is
    simg = StatefulImage.load(
        img_path,
        is_orientation=True,
        to_orientation=None)

    # Convert to world manually
    simg.to_world_direction()
    np.testing.assert_allclose(
        simg.get_fdata()[
            0, 0, 0], [
            0, -1, 0], atol=1e-5)

    # Save in World Space
    world_save_path = str(tmp_path / "world_save.nii.gz")
    simg.save(world_save_path)
    simg_world = StatefulImage.load(
        world_save_path,
        is_orientation=True,
        to_orientation=None)
    np.testing.assert_allclose(
        simg_world.get_fdata()[
            0, 0, 0], [
            0, -1, 0], atol=1e-5)

    # Convert back to Voxel Space
    simg.to_voxel_direction()
    np.testing.assert_allclose(simg.get_fdata()[0, 0, 0], [0, 0, 1], atol=1e-5)

    # Save in Voxel Space
    voxel_save_path = str(tmp_path / "voxel_save.nii.gz")
    simg.save(voxel_save_path)
    simg_voxel = StatefulImage.load(
        voxel_save_path,
        is_orientation=True,
        to_orientation=None)
    np.testing.assert_allclose(
        simg_voxel.get_fdata()[
            0, 0, 0], [
            0, 0, 1], atol=1e-5)


def test_stateful_image_save_reoriented(tmp_path):
    # Test saving when the in-memory image is reoriented
    original_affine = np.eye(4)  # RAS
    data_peaks = np.zeros((2, 2, 2, 3))
    data_peaks[:, :, :, :] = [1, 0, 0]  # X (Right)

    img_path = str(tmp_path / "ras.nii.gz")
    nib.save(nib.Nifti1Image(data_peaks, original_affine), img_path)

    # Load and reorient to LAS
    simg = StatefulImage.load(img_path, to_orientation="LAS",
                              is_orientation=True)

    # Data is not automatically rotated by default
    np.testing.assert_allclose(simg.get_fdata()[0, 0, 0], [1, 0, 0], atol=1e-5)

    # Save back to original (RAS).
    save_world = str(tmp_path / "save_world.nii.gz")
    simg.save(save_world)
    raw_world = nib.load(save_world).get_fdata()
    np.testing.assert_allclose(raw_world[0, 0, 0], [1, 0, 0], atol=1e-5)

    # Let's try if original orientation was LAS
    las_affine = np.diag([-1, 1, 1, 1])
    las_path = str(tmp_path / "las.nii.gz")
    # Voxel [1, 0, 0] in LAS means Left.
    nib.save(nib.Nifti1Image(data_peaks, las_affine), las_path)

    simg_las = StatefulImage.load(las_path, to_orientation=None,
                                  is_orientation=True)
    # Not automatically rotated.
    np.testing.assert_allclose(
        simg_las.get_fdata()[
            0, 0, 0], [
            1, 0, 0], atol=1e-5)

    # We can manually rotate it
    simg_las.to_world_direction()
    np.testing.assert_allclose(
        simg_las.get_fdata()[0, 0, 0], [-1, 0, 0], atol=1e-5)
