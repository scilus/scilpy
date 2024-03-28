#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_dwi_compute_snr.py', '--help')
    assert ret.success


def test_snr(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           'dwi.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           'dwi.bvec')
    in_mask = os.path.join(SCILPY_HOME, 'processing',
                           'cc.nii.gz')
    noise_mask = os.path.join(SCILPY_HOME, 'processing',
                              'small_roi_gm_mask.nii.gz')

    ret = script_runner.run('scil_dwi_compute_snr.py', in_dwi,
                            in_bval, in_bvec, in_mask,
                            '--noise_mask', noise_mask,
                            '--b0_thr', '10')
    assert ret.success
