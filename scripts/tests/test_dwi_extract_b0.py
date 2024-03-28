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
    ret = script_runner.run('scil_dwi_extract_b0.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           'dwi.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           'dwi.bvec')
    ret = script_runner.run('scil_dwi_extract_b0.py', in_dwi, in_bval, in_bvec,
                            'b0_mean.nii.gz', '--mean', '--b0', '20')
    assert ret.success

    # Test wrong b0. Current minimal b-value is 5.
    ret = script_runner.run('scil_dwi_extract_b0.py', in_dwi, in_bval, in_bvec,
                            'b0_mean.nii.gz', '--mean', '--b0', '1', '-f')
    assert not ret.success
