#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip', 'processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_denoising_nlmeans.py', '--help')
    assert ret.success


def test_execution_user_sigma(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(SCILPY_HOME, 'others', 't1_resample.nii.gz')
    ret = script_runner.run('scil_denoising_nlmeans.py', in_img,
                            'denoised.nii.gz', '--processes', '1',
                            '--sigma', '8', '--number_coils', 0,
                            '--gaussian')
    assert ret.success


def test_execution_basic_3d(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(SCILPY_HOME, 'others', 't1_resample.nii.gz')
    ret = script_runner.run('scil_denoising_nlmeans.py', in_img,
                            't1_denoised.nii.gz', '--processes', '1',
                            '--basic_sigma', '--number_coils', 0,
                            '--gaussian')
    assert ret.success


def test_execution_basic_4d_mask(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(SCILPY_HOME, 'processing', 'dwi_crop_1000.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'processing', 'fa_thr.nii.gz')
    ret = script_runner.run('scil_denoising_nlmeans.py', in_img,
                            't1_denoised2.nii.gz', '--processes', '1',
                            '--basic_sigma', '--number_coils', 0,
                            '--gaussian', '--mask_sigma', in_mask)
    assert ret.success


def test_execution_piesno(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(SCILPY_HOME, 'processing', 'dwi.nii.gz')
    ret = script_runner.run('scil_denoising_nlmeans.py', in_img,
                            'dwi_denoised.nii.gz', '--processes', '1',
                            '--piesno', '--number_coils', '4')
    assert ret.success
