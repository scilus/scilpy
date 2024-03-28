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
    ret = script_runner.run('scil_dwi_split_by_indices.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing', 'dwi_crop.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing', 'dwi.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing', 'dwi.bvec')
    ret = script_runner.run('scil_dwi_split_by_indices.py', in_dwi,
                            in_bval, in_bvec, 'dwi', '5', '15', '25')
    assert ret.success


def test_execution_processing_wrong_indices_given(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing', 'dwi_crop.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing', 'dwi.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing', 'dwi.bvec')
    ret = script_runner.run('scil_dwi_split_by_indices.py', in_dwi,
                            in_bval, in_bvec, 'dwi', '0', '15', '25')
    assert (not ret.success)
    ret = script_runner.run('scil_dwi_split_by_indices.py', in_dwi,
                            in_bval, in_bvec, 'dwi', '5', '15', '200')
    assert (not ret.success)
