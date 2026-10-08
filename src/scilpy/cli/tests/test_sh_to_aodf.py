#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from scilpy.gpuparallel.opencl_utils import have_opencl
from scilpy.io.dvc import pull_test_case_package

# If they already exist, this only takes 5 seconds (check md5sum)
test_data_root = pull_test_case_package("aodf")
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_sh_to_aodf', '--help'])
    assert ret.success


@pytest.mark.parametrize("in_fodf,expected_fodf", [
    [os.path.join(test_data_root, "fodf_descoteaux07_sub.nii.gz"),
     os.path.join(test_data_root,
                  "fodf_descoteaux07_sub_unified_asym.nii.gz")]])
def test_asym_basis_output_gpu(script_runner, in_fodf,
                               expected_fodf, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run(['scil_sh_to_aodf',
                             in_fodf, 'out_fodf1.nii.gz',
                             '--sphere', 'repulsion100',
                             '--sigma_align', '0.8',
                             '--sigma_spatial', '1.0',
                             '--sigma_range', '0.2',
                             '--sigma_angle', '0.06',
                             '--use_opencl',
                             '--device', 'gpu',
                             '--sh_basis', 'descoteaux07_legacy', '-f',
                             '--include_center'])

    if have_opencl:
        # if we have opencl the script should not raise an error
        assert ret.success

        # output should be close to expected (but not exactly equal because
        # the python implementation is float64 while gpu is float32)
        ret_fodf = nib.load("out_fodf1.nii.gz")
        test_fodf = nib.load(expected_fodf)
        assert np.allclose(ret_fodf.get_fdata(),
                           test_fodf.get_fdata(),
                           atol=1e-6)
    else:
        # if we don't have opencl the script should have raised an error
        assert not ret.success


@pytest.mark.parametrize("in_fodf,expected_fodf", [
    [os.path.join(test_data_root, "fodf_descoteaux07_sub.nii.gz"),
     os.path.join(test_data_root,
                  "fodf_descoteaux07_sub_unified_asym.nii.gz")]])
def test_asym_basis_output(script_runner, in_fodf, expected_fodf, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run(['scil_sh_to_aodf',
                             in_fodf, 'out_fodf1.nii.gz',
                             '--sphere', 'repulsion100',
                             '--sigma_align', '0.8',
                             '--sigma_spatial', '1.0',
                             '--sigma_range', '0.2',
                             '--sigma_angle', '0.06',
                             '--device', 'cpu',
                             '--sh_basis', 'descoteaux07_legacy', '-f',
                             '--include_center'])

    assert ret.success

    ret_fodf = nib.load("out_fodf1.nii.gz")
    test_fodf = nib.load(expected_fodf)
    assert np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())


@pytest.mark.parametrize("in_fodf,expected_fodf", [
    [os.path.join(test_data_root,
                  "fodf_descoteaux07_sub_unified_asym.nii.gz"),
     os.path.join(test_data_root,
                  "fodf_descoteaux07_sub_unified_asym_twice.nii.gz")]])
def test_asym_input(script_runner, in_fodf, expected_fodf, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run(['scil_sh_to_aodf',
                             in_fodf, 'out_fodf1.nii.gz',
                             '--sphere', 'repulsion100',
                             '--sigma_align', '0.8',
                             '--sigma_spatial', '1.0',
                             '--sigma_range', '0.2',
                             '--sigma_angle', '0.06',
                             '--device', 'cpu',
                             '--sh_basis', 'descoteaux07_legacy', '-f',
                             '--include_center'])

    assert ret.success

    ret_fodf = nib.load("out_fodf1.nii.gz")
    test_fodf = nib.load(expected_fodf)
    assert np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())


@pytest.mark.parametrize("in_fodf,out_fodf", [
    [os.path.join(test_data_root, 'fodf_descoteaux07_sub.nii.gz'),
     os.path.join(test_data_root,
                  'fodf_descoteaux07_sub_cosine_asym.nii.gz')]])
def test_cosine_method(script_runner, in_fodf, out_fodf, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run(['scil_sh_to_aodf',
                             in_fodf, 'out_fodf1.nii.gz',
                             '--sphere', 'repulsion100',
                             '--method', 'cosine', '-f',
                             '--sh_basis', 'descoteaux07_legacy'])

    assert ret.success

    ret_fodf = nib.load("out_fodf1.nii.gz")
    test_fodf = nib.load(out_fodf)

    assert np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())


def test_non_ras_sh_to_aodf(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_fodf = os.path.join(test_data_root, 'fodf_descoteaux07_sub.nii.gz')
    img = nib.load(in_fodf)
    # Use a small crop for fast execution
    data_crop = img.get_fdata(dtype=np.float32)[:7, :7, :7]
    aff_ras = img.affine.copy()
    nib.save(nib.Nifti1Image(data_crop, aff_ras), 'sh_ras.nii.gz')

    aff_las = aff_ras.copy()
    aff_las[0, 0] = -aff_las[0, 0]
    aff_las[0, 3] = aff_las[0, 3] + (data_crop.shape[0] - 1) * 2.5
    data_las = data_crop[::-1].copy()
    nib.save(nib.Nifti1Image(data_las, aff_las), 'sh_las.nii.gz')

    # Run on RAS
    ret_ras = script_runner.run([
        'scil_sh_to_aodf', 'sh_ras.nii.gz', 'out_ras.nii.gz',
        '--sphere', 'repulsion100', '--method', 'cosine', '-f',
        '--sh_basis', 'descoteaux07_legacy'])
    assert ret_ras.success

    # Run on LAS
    ret_las = script_runner.run([
        'scil_sh_to_aodf', 'sh_las.nii.gz', 'out_las.nii.gz',
        '--sphere', 'repulsion100', '--method', 'cosine', '-f',
        '--sh_basis', 'descoteaux07_legacy'])
    assert ret_las.success

    # Verify orientation is preserved
    out_l = nib.load('out_las.nii.gz')
    assert nib.orientations.aff2axcodes(out_l.affine) == ('L', 'A', 'S')

    out_r = nib.load('out_ras.nii.gz')
    assert nib.orientations.aff2axcodes(out_r.affine) == ('R', 'A', 'S')

    # Compare output values
    assert np.allclose(out_r.get_fdata(), out_l.get_fdata()[::-1], atol=1e-5)
