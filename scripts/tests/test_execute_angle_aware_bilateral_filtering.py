#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import os
import pytest
import tempfile
import time


tmp_dir = tempfile.TemporaryDirectory()
cwd = os.getcwd()


@pytest.fixture
def mock_filtering(mocker, out_fodf):
    def _mock(*args, **kwargs):
        img = nib.load(out_fodf)
        return img.get_fdata()

    script = 'scil_execute_angle_aware_bilateral_filtering'
    filtering_fn = "angle_aware_bilateral_filtering"
    return mocker.patch("scripts.{}.{}".format(script, filtering_fn),
                        side_effect=_mock)


def test_help_option(script_runner):
    ret = script_runner.run('scil_execute_angle_aware_bilateral_filtering.py',
                            '--help')
    assert ret.success


@pytest.mark.parametrize("out_fodf",
                         [os.path.join(cwd, "test_fodf.nii.gz")])
def test_asym_basis_output(script_runner, mock_filtering, out_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_execute_angle_aware_bilateral_filtering.py',
                            os.path.join(cwd, 'in_fodf.nii.gz'),
                            'out_fodf1.nii.gz',
                            '--sphere', 'repulsion100',
                            '--sigma_angular', '1.0',
                            '--sigma_spatial', '1.0',
                            '--sigma_range', '1.0',
                            '--sh_basis', 'descoteaux07',
                            '--processes', '1', '-f',
                            print_result=True, shell=True)

    assert ret.success
    mock_filtering.assert_called_once()

    ret_fodf = nib.load("out_fodf1.nii.gz")
    test_fodf = nib.load(out_fodf)
    assert np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())


@pytest.mark.parametrize("out_fodf",
                         [os.path.join(cwd, "test_fodf.nii.gz")])
def test_sym_basis_output(script_runner, mock_filtering, out_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_execute_angle_aware_bilateral_filtering.py',
                            os.path.join(cwd, 'in_fodf.nii.gz'),
                            'out_fodf2.nii.gz',
                            '--out_sym', 'out_sym.nii.gz',
                            '--sphere', 'repulsion100',
                            '--sigma_angular', '1.0',
                            '--sigma_spatial', '1.0',
                            '--sigma_range', '1.0',
                            '--sh_basis', 'descoteaux07',
                            '--processes', '1', '-f',
                            print_result=True, shell=True)

    assert ret.success
    mock_filtering.assert_called_once()
    
    ret_fodf = nib.load("out_sym.nii.gz")
    test_fodf = nib.load(os.path.join(cwd, "test_fodf_sym.nii.gz"))
    assert np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())


@pytest.mark.parametrize("out_fodf",
                         [os.path.join(cwd, "test_fodf2.nii.gz")])
def test_asym_input(script_runner, mock_filtering, out_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_execute_angle_aware_bilateral_filtering.py',
                            os.path.join(cwd, 'test_fodf.nii.gz'),
                            'out_fodf3.nii.gz',
                            '--sphere', 'repulsion100',
                            '--sigma_angular', '1.0',
                            '--sigma_spatial', '1.0',
                            '--sigma_range', '1.0',
                            '--sh_basis', 'descoteaux07',
                            '--processes', '1', '-f',
                            print_result=True, shell=True)

    assert ret.success
    mock_filtering.assert_called_once()
    
    ret_fodf = nib.load("out_fodf3.nii.gz")
    test_fodf = nib.load(out_fodf)
    assert np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())
