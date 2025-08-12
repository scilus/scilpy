#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['commit_amico.zip',
                                           'processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_NODDI_maps', '--help'])
    assert ret.success


def test_execution_commit_amico(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'commit_amico',
                          'dwi.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'commit_amico',
                           'dwi.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'commit_amico',
                           'dwi.bvec')
    mask = os.path.join(SCILPY_HOME, 'commit_amico',
                        'mask.nii.gz')
    ret = script_runner.run(['scil_NODDI_maps', in_dwi,
                             in_bval, in_bvec, '--mask', mask,
                             '--out_dir', 'noddi', '--tol', '30',
                             '--para_diff', '0.0017', '--iso_diff', '0.003',
                             '--lambda1', '0.5', '--lambda2', '0.001',
                             '--processes', '1', '-f'])
    assert ret.success


def test_single_shell_fail(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')
    ret = script_runner.run(['scil_NODDI_maps', in_dwi,
                             in_bval, in_bvec,
                             '--out_dir', 'noddi', '--tol', '30',
                             '--para_diff', '0.0017', '--iso_diff', '0.003',
                             '--lambda1', '0.5', '--lambda2', '0.001',
                             '--processes', '1', '-f'])
    assert not ret.success
