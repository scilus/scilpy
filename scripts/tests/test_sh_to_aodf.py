#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from scilpy.io.dvc import pull_test_case_package

# If they already exist, this only takes 5 seconds (check md5sum)
test_data_root = pull_test_case_package("aodf")
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_sh_to_aodf.py', '--help')
    assert ret.success


@pytest.mark.parametrize("in_fodf,expected_fodf",
    [[os.path.join(test_data_root, "fodf_descoteaux07_sub.nii.gz"),
      os.path.join(test_data_root,
                   "fodf_descoteaux07_sub_unified_asym.nii.gz")]])
def test_asym_basis_output(script_runner, in_fodf, expected_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_sh_to_aodf.py',
                            in_fodf, 'out_fodf1.nii.gz',
                            '--sphere', 'repulsion100',
                            '--sigma_align', '0.8',
                            '--sigma_spatial', '1.0',
                            '--sigma_range', '0.2',
                            '--sigma_angle', '0.06',
                            '--device', 'cpu',
                            '--sh_basis', 'descoteaux07_legacy', '-f',
                            '--include_center',
                            print_result=True, shell=True)

    assert ret.success

    ret_fodf = nib.load("out_fodf1.nii.gz")
    test_fodf = nib.load(expected_fodf)
    assert np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())


@pytest.mark.parametrize("in_fodf,expected_fodf",
    [[os.path.join(test_data_root,
                   "fodf_descoteaux07_sub_unified_asym.nii.gz"),
      os.path.join(test_data_root,
                   "fodf_descoteaux07_sub_unified_asym_twice.nii.gz")]])
def test_asym_input(script_runner, in_fodf, expected_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_sh_to_aodf.py',
                            in_fodf, 'out_fodf1.nii.gz',
                            '--sphere', 'repulsion100',
                            '--sigma_align', '0.8',
                            '--sigma_spatial', '1.0',
                            '--sigma_range', '0.2',
                            '--sigma_angle', '0.06',
                            '--device', 'cpu',
                            '--sh_basis', 'descoteaux07_legacy', '-f',
                            '--include_center',
                            print_result=True, shell=True)

    assert ret.success

    ret_fodf = nib.load("out_fodf1.nii.gz")
    test_fodf = nib.load(expected_fodf)
    assert np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())


@pytest.mark.parametrize("in_fodf,out_fodf",
    [[os.path.join(test_data_root, 'fodf_descoteaux07_sub.nii.gz'),
      os.path.join(test_data_root,
                   'fodf_descoteaux07_sub_cosine_asym.nii.gz')]])
def test_cosine_method(script_runner, in_fodf, out_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_sh_to_aodf.py',
                            in_fodf, 'out_fodf1.nii.gz',
                            '--sphere', 'repulsion100',
                            '--method', 'cosine', '-f',
                            '--sh_basis', 'descoteaux07_legacy',
                            print_result=True, shell=True)

    assert ret.success

    ret_fodf = nib.load("out_fodf1.nii.gz")
    test_fodf = nib.load(out_fodf)

    assert np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())
