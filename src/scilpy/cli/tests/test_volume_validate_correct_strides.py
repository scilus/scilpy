#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_volume_validate_correct_strides', '--help'])
    assert ret.success


def test_execution_processing_basic(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')

    ret = script_runner.run(['scil_volume_validate_correct_strides', in_dwi,
                             'dwi_restride.nii.gz', '-f'])
    assert ret.success


def test_execution_processing_bvecs(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')

    ret = script_runner.run(['scil_volume_validate_correct_strides', in_dwi,
                             'dwi_restride.nii.gz', '--in_bvec', in_bvec,
                             '--out_bvec', 'dwi_restride.bvec', '-f'])
    assert ret.success


def test_execution_processing_bvecs(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')

    ret = script_runner.run(['scil_volume_validate_correct_strides', in_dwi,
                             'dwi_restride.nii.gz', '--in_bvec', in_bvec,
                             '--out_bvec', 'dwi_restride.bvec',
                             '--validate_bvec', '--in_bval', in_bval, '-f'])
    assert ret.success
