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
    ret = script_runner.run('scil_dti_metrics.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing', 'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing', '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing', '1000.bvec')

    # No mask fitting with this data? Creating our own.
    mask = os.path.join(SCILPY_HOME, 'processing', 'ad.nii.gz')
    mask_uint8 = os.path.join('mask_uint8.nii.gz')
    script_runner.run('scil_volume_math.py', 'convert',
                      mask, mask_uint8, '--data_type', 'uint8')

    ret = script_runner.run('scil_dti_metrics.py', in_dwi,
                            in_bval, in_bvec, '--not_all',
                            '--fa', 'fa.nii.gz',
                            '--md', 'md.nii.gz',
                            '--ad', 'ad.nii.gz',
                            '--rd', 'rd.nii.gz',
                            '--residual', 'residual.nii.gz',
                            '--mask', mask_uint8)
    assert ret.success

    ret = script_runner.run('scil_dti_metrics.py', in_dwi,
                            in_bval, in_bvec, '--not_all',
                            '--fa', 'fa.nii.gz', '--b0_threshold', '1', '-f')
    assert not ret.success
