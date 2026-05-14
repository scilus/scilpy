# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import pytest
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.io.stateful_tractogram import Space
from dipy.io.streamline import load_tractogram
from scilpy.io.stateful_image import StatefulImage
from scilpy.tracking.seed import SeedGenerator
from scilpy.tracking.utils import save_tractogram


@pytest.fixture
def rotated_las_dataset(tmp_path):
    """
    Create a mock LAS dataset with 45-degree rotation around Z and 2mm voxels.
    """
    affine = np.array([
        [-1.414, 1.414, 0.0, 50.0],
        [-1.414, -1.414, 0.0, 50.0],
        [0.0, 0.0, 2.0, -20.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Gradients (6 dirs + 1 b0) - Defined in WORLD X alignment
    bvecs_world = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [-1, 0, 0], [0, -1, 0], [0, 0, -1]
    ])
    bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000])

    # Simulate signal
    data = np.ones((10, 10, 10, 7)) * 20
    data[2:8, 2:8, 2:8, 0] = 100
    for i in range(1, 7):
        g = bvecs_world[i]
        cos_theta = np.dot(g, [1, 0, 0])
        data[2:8, 2:8, 2:8, i] = 100 * np.exp(-1.0 * (cos_theta**2))

    # Back-project world bvecs to voxel space for FSL file
    R = affine[:3, :3]
    R_inv = np.linalg.inv(R / np.linalg.norm(R, axis=0))
    bvecs_vox = np.dot(bvecs_world, R_inv.T)
    if np.linalg.det(R) > 0:
        bvecs_vox[:, 0] *= -1

    dwi_path = str(tmp_path / "dwi.nii.gz")
    bval_path = str(tmp_path / "dwi.bval")
    bvec_path = str(tmp_path / "dwi.bvec")

    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), dwi_path)
    np.savetxt(bval_path, bvals[None, :], fmt='%d')
    np.savetxt(bvec_path, bvecs_vox.T, fmt='%.8f')

    return dwi_path, bval_path, bvec_path, bvecs_world


def test_stateful_image_world_gradients(rotated_las_dataset):
    dwi, bval, bvec, bvecs_world_truth = rotated_las_dataset
    simg = StatefulImage.load(dwi)
    simg.load_gradients(bval, bvec)

    # Assert world_bvecs match truth
    np.testing.assert_allclose(simg.world_bvecs, bvecs_world_truth, atol=1e-2)

    # Assert saving and reloading recovers world truth
    tmp_bvec = bvec + "_tmp.bvec"
    simg.save_gradients(bval, tmp_bvec)
    simg2 = StatefulImage.load(dwi)
    simg2.load_gradients(bval, tmp_bvec)
    np.testing.assert_allclose(simg2.world_bvecs, bvecs_world_truth, atol=1e-2)


def test_dti_fitting_world_space(rotated_las_dataset):
    dwi, bval, bvec, _ = rotated_las_dataset
    simg = StatefulImage.load(dwi)
    simg.load_gradients(bval, bvec)

    gtab = gradient_table(simg.bvals, bvecs=simg.world_bvecs)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(simg.get_fdata())

    peak = tenfit.evecs[5, 5, 5, 0]
    # Fiber was simulated along physical X
    assert np.abs(peak[0]) > 0.8


def test_tracking_seeding_world_space(rotated_las_dataset):
    dwi, bval, bvec, _ = rotated_las_dataset
    simg = StatefulImage.load(dwi)

    seed_data = np.zeros((10, 10, 10))
    seed_data[5, 5, 5] = 1

    seed_gen = SeedGenerator(seed_data, simg.header.get_zooms()[:3],
                             affine=simg.affine, space=Space.RASMM)

    rng = np.random.RandomState(42)
    seed = seed_gen.get_next_pos(rng, np.arange(1), 0)

    # Project back to voxel space and check index
    inv_affine = np.linalg.inv(simg.affine)
    seed_vox = np.dot(inv_affine, np.append(seed, 1.0))[:3]
    np.testing.assert_allclose(seed_vox, [5.5, 5.5, 5.5], atol=1.0)


def test_save_tractogram_world_space(tmp_path, rotated_las_dataset):
    dwi, bval, bvec, _ = rotated_las_dataset
    simg = StatefulImage.load(dwi)

    # World coordinate for center of (5,5,5)
    seed_world = np.dot(simg.affine, [5.5, 5.5, 5.5, 1])[:3]
    streamline = np.array([seed_world, seed_world + [10, 0, 0]], dtype=float)

    def mock_gen():
        yield streamline, seed_world

    out_trk_scil = str(tmp_path / "test_scil.trk")
    from nibabel.streamlines import TrkFile as NibTrkFile
    save_tractogram(mock_gen, NibTrkFile, simg, 1, out_trk_scil,
                    0, 1000, None, True, False, space=Space.RASMM)

    sft_scil = load_tractogram(out_trk_scil, dwi, bbox_valid_check=False)
    assert len(sft_scil.streamlines) == 1, "Scilpy save_tractogram produced empty file!"
    sft_scil.to_rasmm()

    # Assert coordinates match
    np.testing.assert_allclose(sft_scil.streamlines[0], streamline, atol=1e-2)
    # Assert seed was saved correctly in DPS
    np.testing.assert_allclose(sft_scil.data_per_streamline['seeds'][0], seed_world, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
