# -*- coding: utf-8 -*-

import os
import pytest
import tempfile
from contextlib import contextmanager

import nibabel as nib
import numpy as np

from scilpy.io.stateful_image import StatefulImage


@contextmanager
def create_dummy_nifti_with_gradients(filename="test.nii.gz", n_volumes=5, affine=None):
    """
    Create a dummy NIfTI file and gradient files for testing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (10, 10, 10, n_volumes)
        if affine is None:
            affine = np.eye(4)
        data = np.random.rand(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, affine)

        file_path = os.path.join(tmpdir, filename)
        nib.save(img, file_path)

        bvals = np.random.randint(0, 3000, n_volumes)
        bvecs = np.random.randn(n_volumes, 3)
        bvecs /= (np.linalg.norm(bvecs, axis=1)[:, None] + 1e-8)

        bval_path = os.path.join(tmpdir, "test.bval")
        bvec_path = os.path.join(tmpdir, "test.bvec")

        np.savetxt(bval_path, bvals[None, :], fmt='%d')
        np.savetxt(bvec_path, bvecs.T, fmt='%.8f')

        yield file_path, bval_path, bvec_path, bvals, bvecs


def test_attach_gradients():
    with create_dummy_nifti_with_gradients() as (img_p, bval_p, bvec_p, bvals, bvecs):
        simg = StatefulImage.load(img_p)
        simg.attach_gradients(bvals, bvecs)

        assert np.allclose(simg.bvals, bvals)
        assert np.allclose(simg.bvecs, bvecs)


def test_load_gradients():
    with create_dummy_nifti_with_gradients() as (img_p, bval_p, bvec_p, bvals, bvecs):
        simg = StatefulImage.load(img_p)
        simg.load_gradients(bval_p, bvec_p)

        assert np.allclose(simg.bvals, bvals)
        assert np.allclose(simg.bvecs, bvecs, atol=1e-5)


def test_reorient_gradients():
    with create_dummy_nifti_with_gradients() as (img_p, bval_p, bvec_p, bvals, bvecs):
        simg = StatefulImage.load(img_p)
        simg.attach_gradients(bvals, bvecs)

        # LPS reorientation: flip x and y
        simg.to_lps()
        assert simg.axcodes == ("L", "P", "S", "T")

        expected_bvecs = bvecs.copy()
        expected_bvecs[:, 0] *= -1
        expected_bvecs[:, 1] *= -1

        assert np.allclose(simg.bvecs, expected_bvecs)

        # Reorient back to RAS
        simg.to_ras()
        assert simg.axcodes == ("R", "A", "S", "T")
        assert np.allclose(simg.bvecs, bvecs)


def test_save_gradients():
    with create_dummy_nifti_with_gradients() as (img_p, bval_p, bvec_p, bvals, bvecs):
        simg = StatefulImage.load(img_p)
        simg.attach_gradients(bvals, bvecs)
        simg.to_lps()

        tmp_dir = os.path.dirname(img_p)
        out_bval = os.path.join(tmp_dir, "out.bval")
        out_bvec = os.path.join(tmp_dir, "out.bvec")

        simg.save_gradients(out_bval, out_bvec)

        # Saved gradients should be back in RAS (original)
        saved_bvals = np.loadtxt(out_bval)
        saved_bvecs = np.loadtxt(out_bvec).T

        assert np.allclose(saved_bvals, bvals)
        assert np.allclose(saved_bvecs, bvecs)

        # StatefulImage itself should still be in LPS
        assert simg.axcodes == ("L", "P", "S", "T")


def test_create_from_with_gradients():
    with create_dummy_nifti_with_gradients() as (img_p, bval_p, bvec_p, bvals, bvecs):
        simg = StatefulImage.load(img_p)
        simg.attach_gradients(bvals, bvecs)
        simg.to_lps()

        # Create new simg from source (RAS) but with same reference (LPS)
        source_nii = nib.load(img_p)
        new_simg = StatefulImage.create_from(source_nii, simg)

        # new_simg matches source_nii (RAS)
        assert new_simg.axcodes == ("R", "A", "S", "T")
        # bvecs should have been reoriented back to RAS to match source_nii
        assert np.allclose(new_simg.bvecs, bvecs)
        assert np.allclose(new_simg.bvals, bvals)


def test_validation_errors():
    with create_dummy_nifti_with_gradients(n_volumes=5) as \
            (img_p, bval_p, bvec_p, bvals, bvecs):
        simg = StatefulImage.load(img_p)

        # Wrong number of volumes
        with pytest.raises(ValueError,
                           match="Number of gradients.*does not match number of volumes"):
            simg.attach_gradients(bvals[:3], bvecs[:3])

        # Wrong shape
        with pytest.raises(ValueError, match="bvals must be a 1D array"):
            simg.attach_gradients(bvals[:, None], bvecs)


def test_gradient_consistency_across_orientations():
    """
    Comprehensive test:
    1. Create RAS image + gradients.
    2. Reorient to LAS, LPS, LPI.
    3. Save in those orientations.
    4. Load back and verify they all return to the same RAS state.
    """
    n_volumes = 4
    with create_dummy_nifti_with_gradients(n_volumes=n_volumes) as \
            (img_p, bval_p, bvec_p, bvals, bvecs):
        simg_ras = StatefulImage.load(img_p)
        simg_ras.attach_gradients(bvals, bvecs)

        # Original bvecs are in RAS (matching simg_ras.axcodes)
        original_bvecs = simg_ras.bvecs.copy()

        for target_ornt in ["LAS", "LPS", "LPI"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                # 1. Reorient
                simg_ras.reorient(target_ornt)

                # 2. Create a "new" original at this orientation so we can save it AS is
                # convert_to_simg sets the current state as the "original"
                simg_target = StatefulImage.convert_to_simg(
                    simg_ras, simg_ras.bvals, simg_ras.bvecs)

                # 3. Save
                target_img_p = os.path.join(tmpdir, "target.nii.gz")
                target_bval_p = os.path.join(tmpdir, "target.bval")
                target_bvec_p = os.path.join(tmpdir, "target.bvec")

                simg_target.save(target_img_p)
                simg_target.save_gradients(target_bval_p, target_bvec_p)

                # 4. Load back (defaults to RAS)
                simg_verify = StatefulImage.load(
                    target_img_p, to_orientation="RAS")
                simg_verify.load_gradients(target_bval_p, target_bvec_p)

                # 5. Verify
                assert simg_verify.axcodes == ("R", "A", "S", "T")
                # Threshold for float precision after multiple transforms
                assert np.allclose(simg_verify.bvecs,
                                   original_bvecs, atol=1e-5)
                assert np.allclose(simg_verify.bvals, bvals)

                # Go back to RAS for next iteration
                simg_ras.to_ras()


def test_world_bvecs_non_diagonal_affine():
    # Create a rotation matrix (45 degrees around Z)
    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    affine = np.eye(4)
    affine[:3, :3] = R

    with create_dummy_nifti_with_gradients(affine=affine) as (img_p, bval_p, bvec_p, bvals, bvecs):
        simg = StatefulImage.load(img_p)
        simg.attach_gradients(bvals, bvecs)

        # world_bvecs should be (bvecs * [-1, 1, 1]) * R.T because det(R) > 0
        bvecs_flipped = bvecs.copy()
        bvecs_flipped[:, 0] *= -1
        expected_world_bvecs = np.dot(bvecs_flipped, R.T)
        assert np.allclose(simg.world_bvecs, expected_world_bvecs)

        # bvecs property should return original bvecs (voxel space)
        assert np.allclose(simg.bvecs, bvecs)

        # Save and reload
        tmp_dir = os.path.dirname(img_p)
        out_bval = os.path.join(tmp_dir, "out.bval")
        out_bvec = os.path.join(tmp_dir, "out.bvec")
        simg.save_gradients(out_bval, out_bvec)

        saved_bvecs = np.loadtxt(out_bvec).T
        assert np.allclose(saved_bvecs, bvecs)


def test_world_bvecs_negative_det_affine():
    # LAS affine (det < 0)
    affine = np.diag([-1, 1, 1, 1])

    with create_dummy_nifti_with_gradients(affine=affine) as (img_p, bval_p, bvec_p, bvals, bvecs):
        simg = StatefulImage.load(img_p)
        simg.attach_gradients(bvals, bvecs)

        # world_bvecs should be bvecs * R.T because det(R) < 0
        R = simg._get_rotation_matrix(affine)
        expected_world_bvecs = np.dot(bvecs, R.T)
        assert np.allclose(simg.world_bvecs, expected_world_bvecs)

        # Verify that bvecs property returns original bvecs
        assert np.allclose(simg.bvecs, bvecs)


def test_world_bvecs_reorientation_roundtrip():
    # Start with LAS (det < 0)
    affine_las = np.diag([-1, 1, 1, 1])

    with create_dummy_nifti_with_gradients(affine=affine_las) as (img_p, bval_p, bvec_p, bvals, bvecs_las):
        simg = StatefulImage.load(img_p)
        simg.attach_gradients(bvals, bvecs_las)

        # Reorient to RAS (det > 0)
        simg.to_ras()
        assert simg.axcodes == ("R", "A", "S", "T")

        # world_bvecs should remain the same
        R_las = simg._get_rotation_matrix(affine_las)
        expected_world_bvecs = np.dot(bvecs_las, R_las.T)
        assert np.allclose(simg.world_bvecs, expected_world_bvecs)

        # bvecs in RAS should be flipped in X compared to world_bvecs * R_ras
        # because det(RAS) > 0.
        # Since R_ras = I, bvecs_ras = world_bvecs * [-1, 1, 1]
        expected_bvecs_ras = expected_world_bvecs.copy()
        expected_bvecs_ras[:, 0] *= -1
        assert np.allclose(simg.bvecs, expected_bvecs_ras)

        # Reorient back to LAS
        simg.reorient("LAS")
        assert np.allclose(simg.bvecs, bvecs_las)
        assert np.allclose(simg.world_bvecs, expected_world_bvecs)
