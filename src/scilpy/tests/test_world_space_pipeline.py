import os
import numpy as np
import nibabel as nib
import pytest
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table

from scilpy.io.stateful_image import StatefulImage

def test_world_space_pipeline(tmp_path):
    # 1. Generate mock dataset with 45 degree rotation around Z
    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    affine = np.eye(4)
    affine[:3, :3] = R
    
    shape = (10, 10, 10)
    n_volumes = 7 # 1 b0 + 6 directions
    data = np.ones(shape + (n_volumes,))
    
    # Create a synthetic DTI signal: a single fiber along X in world space
    # In voxel space, this fiber should be along R.T * [1, 0, 0]
    # Because v_world = R * v_vox => v_vox = R.T * v_world
    fiber_dir_world = np.array([1, 0, 0])
    fiber_dir_vox = np.dot(R.T, fiber_dir_world)
    
    bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000])
    # Directions in voxel space
    bvecs_vox = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=float)
    norms = np.linalg.norm(bvecs_vox, axis=1)
    bvecs_vox[norms > 0] /= norms[norms > 0][:, None]
    
    # Simple DTI signal simulation
    # S = S0 * exp(-b * (g.T * D * g))
    # For a single fiber along fiber_dir_vox: D = l1 * v*v.T + l2 * (I - v*v.T)
    l1, l2 = 1.5e-3, 0.5e-3
    V = fiber_dir_vox[:, None]
    D = l1 * np.dot(V, V.T) + l2 * (np.eye(3) - np.dot(V, V.T))
    
    for i in range(n_volumes):
        if bvals[i] == 0:
            data[..., i] = 100
        else:
            g = bvecs_vox[i]
            data[..., i] = 100 * np.exp(-bvals[i] * np.dot(g, np.dot(D, g)))
            
    img_path = str(tmp_path / "data.nii.gz")
    nib.save(nib.Nifti1Image(data, affine), img_path)
    
    bval_path = str(tmp_path / "data.bval")
    bvec_path = str(tmp_path / "data.bvec")
    np.savetxt(bval_path, bvals[None, :], fmt='%d')
    np.savetxt(bvec_path, bvecs_vox.T, fmt='%.8f')
    
    # 2. Load using StatefulImage
    simg = StatefulImage.load(img_path)
    simg.load_gradients(bval_path, bvec_path)
    
    # 3. DTI Fit
    # Use dipy directly but with simg data and gradients
    gtab = gradient_table(simg.bvals, bvecs=simg.bvecs) # simg.bvecs are in voxel space
    
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(simg.get_fdata())
    
    # 4. Peak Extraction
    # The principal eigenvector (V1) should be along fiber_dir_vox in voxel space
    v1 = tenfit.evecs[5, 5, 5, :, 0]
    # Ensure it's pointing in the same hemisphere
    if np.dot(v1, fiber_dir_vox) < 0:
        v1 = -v1
    assert np.allclose(v1, fiber_dir_vox, atol=1e-2)
    
    # 5. Tracking
    # Simple tracking: just follow V1
    streamline = [np.array([
        [5, 5, 5],
        [5, 5, 5] + v1,
        [5, 5, 5] + 2*v1
    ])]
    
    sft = StatefulTractogram(streamline, simg, Space.VOX)
    
    # 6. Save
    tract_path = str(tmp_path / "tract.trk")
    save_tractogram(sft, tract_path)
    
    # 7. Assertions
    # Reload and check world space coordinates
    sft_loaded = load_tractogram(tract_path, img_path)
    sft_loaded.to_rasmm()
    
    # The streamline in world space should be along fiber_dir_world
    # Start point in world space:
    start_vox = np.array([5, 5, 5, 1])
    start_world = np.dot(affine, start_vox)[:3]
    
    loaded_world = sft_loaded.streamlines[0]
    
    # Direction in world space
    dir_world = loaded_world[1] - loaded_world[0]
    dir_world /= np.linalg.norm(dir_world)
    
    if np.dot(dir_world, fiber_dir_world) < 0:
        dir_world = -dir_world

    assert np.allclose(loaded_world[0], start_world, atol=1e-3)
    assert np.allclose(dir_world, fiber_dir_world, atol=1e-2)


def test_world_space_pipeline_negative_det(tmp_path):
    # 1. Generate mock dataset with LAS affine (det < 0)
    affine = np.diag([-1, 1, 1, 1])
    affine[:3, 3] = [50, 50, 50]  # Some translation

    shape = (10, 10, 10)
    n_volumes = 7
    data = np.ones(shape + (n_volumes,))

    # Fiber along X in world space (Right)
    fiber_dir_world = np.array([1, 0, 0])
    # In voxel space (LAS): v_vox = R.T * v_world = [-1, 0, 0]
    fiber_dir_vox = np.array([-1, 0, 0])

    bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000])
    # Directions in voxel space
    bvecs_vox = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=float)
    norms = np.linalg.norm(bvecs_vox, axis=1)
    bvecs_vox[norms > 0] /= norms[norms > 0][:, None]

    # DTI signal simulation
    l1, l2 = 1.5e-3, 0.5e-3
    V = fiber_dir_vox[:, None]
    D = l1 * np.dot(V, V.T) + l2 * (np.eye(3) - np.dot(V, V.T))

    for i in range(n_volumes):
        if bvals[i] == 0:
            data[..., i] = 100
        else:
            g = bvecs_vox[i]
            data[..., i] = 100 * np.exp(-bvals[i] * np.dot(g, np.dot(D, g)))

    img_path = str(tmp_path / "data_las.nii.gz")
    nib.save(nib.Nifti1Image(data, affine), img_path)

    bval_path = str(tmp_path / "data_las.bval")
    bvec_path = str(tmp_path / "data_las.bvec")
    np.savetxt(bval_path, bvals[None, :], fmt='%d')
    np.savetxt(bvec_path, bvecs_vox.T, fmt='%.8f')

    # 2. Load using StatefulImage, keeping original orientation (LAS)
    simg = StatefulImage.load(img_path, to_orientation=None)
    simg.load_gradients(bval_path, bvec_path)

    # 3. DTI Fit
    gtab = gradient_table(simg.bvals, bvecs=simg.bvecs)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(simg.get_fdata())

    # 4. Peak Extraction
    v1 = tenfit.evecs[5, 5, 5, :, 0]
    if np.dot(v1, fiber_dir_vox) < 0:
        v1 = -v1
    assert np.allclose(v1, fiber_dir_vox, atol=1e-2)

    # 5. Tracking
    streamline = [np.array([
        [5, 5, 5],
        [5, 5, 5] + v1,
        [5, 5, 5] + 2*v1
    ])]
    sft = StatefulTractogram(streamline, simg, Space.VOX)

    # 6. Save
    tract_path = str(tmp_path / "tract_las.trk")
    save_tractogram(sft, tract_path)

    # 7. Assertions
    sft_loaded = load_tractogram(tract_path, img_path)
    sft_loaded.to_rasmm()

    start_vox = np.array([5, 5, 5, 1])
    start_world = np.dot(affine, start_vox)[:3]

    loaded_world = sft_loaded.streamlines[0]
    dir_world = loaded_world[1] - loaded_world[0]
    dir_world /= np.linalg.norm(dir_world)

    if np.dot(dir_world, fiber_dir_world) < 0:
        dir_world = -dir_world

    assert np.allclose(loaded_world[0], start_world, atol=1e-3)
    assert np.allclose(dir_world, fiber_dir_world, atol=1e-2)

