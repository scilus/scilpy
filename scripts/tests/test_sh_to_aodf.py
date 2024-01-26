#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import os
import pytest
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home
from scilpy.tests.checks import assert_images_close, assert_images_not_close


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['fodf_filtering.zip'])
data_path = os.path.join(get_home(), 'fodf_filtering')
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_sh_to_aodf.py', '--help')
    assert ret.success


@pytest.mark.parametrize("in_fodf,expected_results",
    [[os.path.join(data_path, 'fodf_descoteaux07_sub.nii.gz'),
      os.path.join(data_path, 'fodf_descoteaux07_sub_full.nii.gz')]],
    scope='function')
def test_asym_basis_output(script_runner, in_fodf, expected_results,
                           mock_collector):

    os.chdir(os.path.expanduser(tmp_dir.name))
    _mocks = mock_collector(["bilateral_filtering"], "scripts.scil_sh_to_aodf")

    ret = script_runner.run('scil_sh_to_aodf.py',
                            in_fodf, 'out_fodf1.nii.gz',
                            '--sphere', 'repulsion100',
                            '--sigma_angular', '1.0',
                            '--sigma_spatial', '1.0',
                            '--sigma_range', '1.0',
                            '--sh_basis', 'descoteaux07', '-f',
                            print_result=True, shell=True)

    assert ret.success

    if _mocks["bilateral_filtering"]:
        _mocks["bilateral_filtering"].assert_called_once()

    assert_images_close(nib.load(expected_results),
                        nib.load("out_fodf1.nii.gz"))


@pytest.mark.parametrize("in_fodf,expected_results,sym_fodf",
    [[os.path.join(data_path, "fodf_descoteaux07_sub.nii.gz"),
      os.path.join(data_path, "fodf_descoteaux07_sub_full.nii.gz"),
      os.path.join(data_path, "fodf_descoteaux07_sub_sym.nii.gz")]],
    scope='function')
def test_sym_basis_output(script_runner, in_fodf, expected_results, sym_fodf,
                          mock_collector):

    os.chdir(os.path.expanduser(tmp_dir.name))
    _mocks = mock_collector(["bilateral_filtering"], "scripts.scil_sh_to_aodf")

    ret = script_runner.run('scil_sh_to_aodf.py',
                            in_fodf,
                            'out_fodf2.nii.gz',
                            '--out_sym', 'out_sym.nii.gz',
                            '--sphere', 'repulsion100',
                            '--sigma_angular', '1.0',
                            '--sigma_spatial', '1.0',
                            '--sigma_range', '1.0',
                            '--sh_basis', 'descoteaux07', '-f',
                            print_result=True, shell=True)

    assert ret.success

    if _mocks["bilateral_filtering"]:
        _mocks["bilateral_filtering"].assert_called_once()

    assert_images_close(nib.load(sym_fodf), nib.load("out_sym.nii.gz"))


@pytest.mark.parametrize("in_fodf,expected_results",
    [[os.path.join(data_path, "fodf_descoteaux07_sub_full.nii.gz"),
      os.path.join(data_path, "fodf_descoteaux07_sub_twice.nii.gz")]],
    scope='function')
def test_asym_input(script_runner, in_fodf, expected_results, mock_collector):

    os.chdir(os.path.expanduser(tmp_dir.name))
    _mocks = mock_collector(["bilateral_filtering"], "scripts.scil_sh_to_aodf")

    ret = script_runner.run('scil_sh_to_aodf.py',
                            in_fodf,
                            'out_fodf3.nii.gz',
                            '--sphere', 'repulsion100',
                            '--sigma_angular', '1.0',
                            '--sigma_spatial', '1.0',
                            '--sigma_range', '1.0',
                            '--sh_basis', 'descoteaux07', '-f',
                            print_result=True, shell=True)

    assert ret.success

    if _mocks["bilateral_filtering"]:
        _mocks["bilateral_filtering"].assert_called_once()

    assert_images_close(nib.load(expected_results),
                        nib.load("out_fodf3.nii.gz"))


@pytest.mark.parametrize("in_fodf,expected_results",
    [[os.path.join(data_path, 'fodf_descoteaux07_sub.nii.gz'),
      os.path.join(data_path, 'fodf_descoteaux07_sub_full.nii.gz')]])
def test_cosine_method(script_runner, in_fodf, expected_results):

    os.chdir(os.path.expanduser(tmp_dir.name))
    # method cosine is fast and not mocked

    ret = script_runner.run('scil_sh_to_aodf.py',
                            in_fodf, 'out_fodf1.nii.gz',
                            '--sphere', 'repulsion100',
                            '--method', 'cosine',
                            '--sh_basis', 'descoteaux07', '-f',
                            print_result=True, shell=True)

    assert ret.success

    # We expect the output to be different from the
    # one obtained with angle-aware bilateral filtering
    assert_images_not_close(nib.load(expected_results),
                            nib.load("out_fodf1.nii.gz"))
