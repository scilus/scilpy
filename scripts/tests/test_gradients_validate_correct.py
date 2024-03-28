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
    ret = script_runner.run('scil_gradients_validate_correct.py', '--help')
    assert ret.success


def test_execution_processing_dti_peaks(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')

    # generate the peaks file and fa map we'll use to test our script
    script_runner.run('scil_dti_metrics.py', in_dwi, in_bval, in_bvec,
                      '--not_all', '--fa', 'fa.nii.gz',
                      '--evecs', 'evecs.nii.gz')
    # test the actual script
    ret = script_runner.run('scil_gradients_validate_correct.py', in_bvec,
                            'evecs_v1.nii.gz', 'fa.nii.gz', 'bvec_corr', '-v')
    assert ret.success


def test_execution_processing_fodf_peaks(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           'dwi.bvec')
    in_peaks = os.path.join(SCILPY_HOME, 'processing',
                            'peaks.nii.gz')
    in_fa = os.path.join(SCILPY_HOME, 'processing',
                         'fa.nii.gz')

    # test the actual script
    ret = script_runner.run('scil_gradients_validate_correct.py', in_bvec,
                            in_peaks, in_fa, 'bvec_corr_fodf', '-v')
    assert ret.success
