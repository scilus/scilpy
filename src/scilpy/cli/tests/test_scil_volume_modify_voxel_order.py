#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import nibabel as nib
import numpy as np
import tempfile


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
