#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import os
import pytest
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['fodf_filtering.zip'])
data_path = os.path.join(get_home(), 'fodf_filtering')
tmp_dir = tempfile.TemporaryDirectory()


@pytest.fixture
def mock_filtering(mocker, out_fodf):
    def _mock(*args, **kwargs):
        img = nib.load(out_fodf)
        return img.get_fdata().astype(np.float32)

    script = 'scil_sh_to_aodf'
    filtering_fn = "angle_aware_bilateral_filtering"
    return mocker.patch("scripts.{}.{}".format(script, filtering_fn),
                        side_effect=_mock, create=True)


def test_help_option(script_runner):
    ret = script_runner.run('scil_sh_to_aodf.py', '--help')
    assert ret.success


@pytest.mark.parametrize("in_fodf,out_fodf",
    [[os.path.join(data_path, 'fodf_descoteaux07_sub.nii.gz'),
      os.path.join(data_path, 'fodf_descoteaux07_sub_full.nii.gz')]])
def test_asym_basis_output(script_runner, mock_filtering, in_fodf, out_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_sh_to_aodf.py',
                            in_fodf, 'out_fodf1.nii.gz',
                            '--sphere', 'repulsion100',
                            '--sigma_angular', '1.0',
                            '--sigma_spatial', '1.0',
                            '--sigma_range', '1.0',
                            '--sh_basis', 'descoteaux07', '-f',
                            print_result=True, shell=True)

    assert ret.success
    mock_filtering.assert_called_once()

    ret_fodf = nib.load("out_fodf1.nii.gz")
    test_fodf = nib.load(out_fodf)
    assert np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())


@pytest.mark.parametrize("in_fodf,out_fodf,sym_fodf",
    [[os.path.join(data_path, "fodf_descoteaux07_sub.nii.gz"),
      os.path.join(data_path, "fodf_descoteaux07_sub_full.nii.gz"),
      os.path.join(data_path, "fodf_descoteaux07_sub_sym.nii.gz")]])
def test_sym_basis_output(
    script_runner, mock_filtering, in_fodf, out_fodf, sym_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

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
    mock_filtering.assert_called_once()

    ret_sym_fodf = nib.load("out_sym.nii.gz")
    test_sym_fodf = nib.load(sym_fodf)
    assert np.allclose(ret_sym_fodf.get_fdata(), test_sym_fodf.get_fdata())


@pytest.mark.parametrize("in_fodf,out_fodf",
    [[os.path.join(data_path, "fodf_descoteaux07_sub_full.nii.gz"),
      os.path.join(data_path, "fodf_descoteaux07_sub_twice.nii.gz")]])
def test_asym_input(script_runner, mock_filtering, in_fodf, out_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

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
    mock_filtering.assert_called_once()
    
    ret_fodf = nib.load("out_fodf3.nii.gz")
    test_fodf = nib.load(out_fodf)
    assert np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())


@pytest.mark.parametrize("in_fodf,out_fodf",
    [[os.path.join(data_path, 'fodf_descoteaux07_sub.nii.gz'),
      os.path.join(data_path, 'fodf_descoteaux07_sub_full.nii.gz')]])
def test_cosine_method(script_runner, mock_filtering, in_fodf, out_fodf):
    os.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_sh_to_aodf.py',
                            in_fodf, 'out_fodf1.nii.gz',
                            '--sphere', 'repulsion100',
                            '--method', 'cosine',
                            '--sh_basis', 'descoteaux07',
                            '-f',
                            print_result=True, shell=True)

    assert ret.success

    # method cosine is fast and not mocked
    mock_filtering.assert_not_called()

    ret_fodf = nib.load("out_fodf1.nii.gz")
    test_fodf = nib.load(out_fodf)

    # We expect the output to be different from the
    # one obtained with angle-aware bilateral filtering
    assert not np.allclose(ret_fodf.get_fdata(), test_fodf.get_fdata())
