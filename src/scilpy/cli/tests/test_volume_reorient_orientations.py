#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import numpy as np
import nibabel as nib

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_volume_reorient_orientations', '--help'])
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    
    # Create a test volume (Peaks) with shape (2, 2, 2, 3)
    # We will use an affine that requires rotation
    affine = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    data_peaks = np.zeros((2, 2, 2, 3))
    data_peaks[:, :, :, :] = [0, 0, 1]  # Voxel Z direction

    in_img = 'in.nii.gz'
    nib.save(nib.Nifti1Image(data_peaks, affine), in_img)

    # 1. Test converting to world space
    out_world = 'out_world.nii.gz'
    ret = script_runner.run([
        'scil_volume_reorient_orientations',
        in_img,
        out_world,
        '--to_world',
        '-f'
    ])
    assert ret.success

    # Load it and verify data has been rotated
    # R = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    # R * [0, 0, 1] = [0, -1, 0]
    world_data = nib.load(out_world).get_fdata()
    np.testing.assert_allclose(world_data[0, 0, 0], [0, -1, 0], atol=1e-5)

    # 2. Test converting from world space back to voxel space
    out_voxel = 'out_voxel.nii.gz'
    ret = script_runner.run([
        'scil_volume_reorient_orientations',
        out_world,
        out_voxel,
        '--to_voxel',
        '-f'
    ])
    assert ret.success

    voxel_data = nib.load(out_voxel).get_fdata()
    np.testing.assert_allclose(voxel_data[0, 0, 0], [0, 0, 1], atol=1e-5)


def test_execution_real_data(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_image = os.path.join(SCILPY_HOME, 'processing', 'peaks.nii.gz')
    
    if os.path.exists(in_image):
        out_world = 'real_world.nii.gz'
        ret = script_runner.run([
            'scil_volume_reorient_orientations',
            in_image,
            out_world,
            '--to_world',
            '-f'
        ])
        assert ret.success
        assert os.path.exists(out_world)

