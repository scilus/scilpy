#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_qball_metrics.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')
    ret = script_runner.run('scil_qball_metrics.py', in_dwi,
                            in_bval, in_bvec)
    assert ret.success


def test_execution_not_all(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')
    ret = script_runner.run('scil_qball_metrics.py', in_dwi,
                            in_bval, in_bvec, "--not_all", "--sh", "2.nii.gz")
    assert ret.success

    # Test wrong b0. Current minimal b-val is 5.
    ret = script_runner.run('scil_qball_metrics.py', in_dwi,
                            in_bval, in_bvec, "--not_all", "--sh", "2.nii.gz",
                            '--b0_threshold', '1', '-f')
    assert not ret.success
