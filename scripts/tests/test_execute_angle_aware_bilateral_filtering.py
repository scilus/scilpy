#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import os
import pytest
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home
from scilpy.tests.checks import assert_images_close
from scilpy.tests.checks import assert_images_close


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['fodf_filtering.zip'])
data_path = os.path.join(get_home(), 'fodf_filtering')
tmp_dir = tempfile.TemporaryDirectory()


@pytest.fixture(scope='function')
def filter_mock(mocker, apply_mocks, out_fodf):
    if apply_mocks:
        def _mock_side_effect(*args, **kwargs):
            img = nib.load(out_fodf)
            return img.get_fdata(dtype=np.float32)

        return mocker.patch(
            "{}.{}".format(
                "scripts.scil_execute_angle_aware_bilateral_filtering",
                "angle_aware_bilateral_filtering"),
            side_effect=_mock_side_effect, create=True)

    return None


def test_help_option(script_runner):
    ret = script_runner.run('scil_execute_angle_aware_bilateral_filtering.py',
                            '--help')
    assert ret.success


@pytest.mark.parametrize("in_fodf,out_fodf",
    [[os.path.join(data_path, 'fodf_descoteaux07_sub.nii.gz'),
      os.path.join(data_path, 'fodf_descoteaux07_sub_full.nii.gz')]],
    scope='function')
def test_asym_basis_output(
    script_runner, filter_mock, apply_mocks, in_fodf, out_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_execute_angle_aware_bilateral_filtering.py',
                            in_fodf,
                            'out_fodf1.nii.gz',
                            '--sphere', 'repulsion100',
                            '--sigma_angular', '1.0',
                            '--sigma_spatial', '1.0',
                            '--sigma_range', '1.0',
                            '--sh_basis', 'descoteaux07',
                            '--processes', '1', '-f',
                            print_result=True, shell=True)

    assert ret.success

    if apply_mocks:
        filter_mock.assert_called_once()

    assert_images_close(nib.load(out_fodf), nib.load("out_fodf1.nii.gz"))


@pytest.mark.parametrize("in_fodf,out_fodf,sym_fodf",
    [[os.path.join(data_path, "fodf_descoteaux07_sub.nii.gz"),
      os.path.join(data_path, "fodf_descoteaux07_sub_full.nii.gz"),
      os.path.join(data_path, "fodf_descoteaux07_sub_sym.nii.gz")]],
    scope='function')
def test_sym_basis_output(
    script_runner, filter_mock, apply_mocks, in_fodf, out_fodf, sym_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_execute_angle_aware_bilateral_filtering.py',
                            in_fodf,
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

    if apply_mocks:
        filter_mock.assert_called_once()

    assert_images_close(nib.load(sym_fodf), nib.load("out_sym.nii.gz"))


@pytest.mark.parametrize("in_fodf,out_fodf",
    [[os.path.join(data_path, "fodf_descoteaux07_sub_full.nii.gz"),
      os.path.join(data_path, "fodf_descoteaux07_sub_twice.nii.gz")]],
    scope='function')
def test_asym_input(script_runner, filter_mock, apply_mocks, in_fodf, out_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_execute_angle_aware_bilateral_filtering.py',
                            in_fodf,
                            'out_fodf3.nii.gz',
                            '--sphere', 'repulsion100',
                            '--sigma_angular', '1.0',
                            '--sigma_spatial', '1.0',
                            '--sigma_range', '1.0',
                            '--sh_basis', 'descoteaux07',
                            '--processes', '1', '-f',
                            print_result=True, shell=True)

    assert ret.success

    if apply_mocks:
        filter_mock.assert_called_once()

    assert_images_close(nib.load(out_fodf), nib.load("out_fodf3.nii.gz"))
