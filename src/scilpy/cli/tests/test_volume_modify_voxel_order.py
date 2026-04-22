#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import nibabel as nib
import numpy as np
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_volume_modify_voxel_order', '--help'])
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_file = 'input.nii.gz'
    img = nib.Nifti1Image(np.zeros((10, 20, 30)), np.eye(4))
    nib.save(img, in_file)

    # Test with character-based voxel order
    out_file_lps = 'output_lps.nii.gz'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_file,
                             out_file_lps, '--new_voxel_order=LPS', '-f'])
    assert ret.success
    lps_img = nib.load(out_file_lps)
    assert nib.aff2axcodes(lps_img.affine) == ('L', 'P', 'S')

    # Test with numeric voxel order
    out_file_asr = 'output_asr.nii.gz'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_file,
                             out_file_asr, '--new_voxel_order=3,1,2', '-f'])
    assert ret.success
    asr_img = nib.load(out_file_asr)
    assert nib.aff2axcodes(asr_img.affine) == ('S', 'R', 'A')

    # Test with negative numeric voxel order
    out_file_lai = 'output_lai.nii.gz'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_file,
                             out_file_lai, '--new_voxel_order=-1,2,-3',
                             '-f'])
    assert ret.success
    lai_img = nib.load(out_file_lai)
    assert nib.aff2axcodes(lai_img.affine) == ('L', 'A', 'I')

    # Test with invalid input
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_file,
                             'output.nii.gz', '--new_voxel_order=invalid',
                             '-f'])
    assert not ret.success


def test_execution_with_gradients(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # 1. Create a 4D dummy NIfTI (RAS)
    n_volumes = 2
    in_file = 'input_4d.nii.gz'
    data = np.zeros((10, 10, 10, n_volumes))
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, in_file)

    # 2. Create bvecs
    bvecs = np.array([[0, 0, 0], [1, 0, 0]])  # X-direction in RAS

    in_bvec = 'input.bvec'
    np.savetxt(in_bvec, bvecs.T, fmt='%.8f')

    # 3. Run script to modify voxel order to LPS
    out_file = 'output_lps.nii.gz'
    out_bvec = 'output_lps.bvec'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_file,
                             out_file, '--new_voxel_order=LPS',
                             '--in_bvec', in_bvec, '--out_bvec', out_bvec, '-f'])
    assert ret.success

    # 4. Verify image
    lps_img = nib.load(out_file)
    assert nib.aff2axcodes(lps_img.affine) == ('L', 'P', 'S')

    # 5. Verify gradients (they should be reoriented to match LPS)
    assert os.path.exists(out_bvec)

    saved_bvecs = np.loadtxt(out_bvec).T  # loadtxt returns (3, N) for FSL

    # RAS to LPS: flip X and Y.
    # Original bvec [1, 0, 0] (X) should become [-1, 0, 0]
    expected_bvecs = np.array([[0, 0, 0], [-1, 0, 0]])
    assert np.allclose(saved_bvecs, expected_bvecs)


def test_execution_with_gradients_numeric(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # 1. Create a 4D dummy NIfTI (RAS)
    n_volumes = 2
    in_file = 'input_4d_num.nii.gz'
    data = np.zeros((10, 10, 10, n_volumes))
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, in_file)

    # 2. Create bvecs
    bvecs = np.array([[0, 0, 0], [1, 0, 0]])  # X-direction in RAS

    in_bvec = 'input_num.bvec'
    np.savetxt(in_bvec, bvecs.T, fmt='%.8f')

    # 3. Run script to modify voxel order to LPS using numeric: -1,-2,3
    out_file = 'output_lps_num.nii.gz'
    out_bvec = 'output_lps_num.bvec'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_file,
                             out_file, '--new_voxel_order=-1,-2,3',
                             '--in_bvec', in_bvec, '--out_bvec', out_bvec, '-f'])
    assert ret.success

    # 4. Verify image
    lps_img = nib.load(out_file)
    assert nib.aff2axcodes(lps_img.affine)[:3] == ('L', 'P', 'S')

    # 5. Verify gradients
    assert os.path.exists(out_bvec)
    saved_bvecs = np.loadtxt(out_bvec).T
    expected_bvecs = np.array([[0, 0, 0], [-1, 0, 0]])
    assert np.allclose(saved_bvecs, expected_bvecs)


def test_execution_real_data(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_image = os.path.join(SCILPY_HOME, 'processing',
                            'dwi_crop.nii.gz')

    # Verify original orientation is RAS
    img_in = nib.load(in_image)
    assert nib.aff2axcodes(img_in.affine) == ('R', 'A', 'S')

    # Test LPS
    out_lps = 'real_lps.nii.gz'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_image,
                             out_lps, '--new_voxel_order=LPS', '-f'])
    assert ret.success
    img = nib.load(out_lps)
    assert nib.aff2axcodes(img.affine) == ('L', 'P', 'S')

    # Test RAS
    out_ras = 'real_ras.nii.gz'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_image,
                             out_ras, '--new_voxel_order=RAS', '-f'])
    assert ret.success
    img = nib.load(out_ras)
    assert nib.aff2axcodes(img.affine) == ('R', 'A', 'S')

    # Test LPI
    out_lpi = 'real_lpi.nii.gz'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_image,
                             out_lpi, '--new_voxel_order=LPI', '-f'])
    assert ret.success
    img = nib.load(out_lpi)
    assert nib.aff2axcodes(img.affine) == ('L', 'P', 'I')


def test_execution_with_bvec_real_data(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_image = os.path.join(SCILPY_HOME, 'processing',
                            'dwi_crop.nii.gz')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           'dwi.bvec')

    # Verify original orientation is RAS
    img_in = nib.load(in_image)
    assert nib.aff2axcodes(img_in.affine) == ('R', 'A', 'S')

    # Test LPI
    out_lpi = 'real_lpi_grad.nii.gz'
    out_bvec = 'real_lpi_grad.bvec'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_image,
                             out_lpi, '--new_voxel_order=LPI',
                             '--in_bvec', in_bvec, '--out_bvec', out_bvec, '-f'])
    assert ret.success

    # Verify image
    img = nib.load(out_lpi)
    assert nib.aff2axcodes(img.affine)[:3] == ('L', 'P', 'I')

    # Verify bvec
    assert os.path.exists(out_bvec)
    old_bvecs = np.loadtxt(in_bvec)
    new_bvecs = np.loadtxt(out_bvec)

    # RAS to LPI: flip X, Y, Z
    assert np.allclose(new_bvecs, -old_bvecs)
