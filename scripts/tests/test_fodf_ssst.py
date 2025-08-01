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
    ret = script_runner.run(['scil_fodf_ssst.py', '--help'])
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_3000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '3000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '3000.bvec')
    in_frf = os.path.join(SCILPY_HOME, 'processing',
                          'frf.txt')
    ret = script_runner.run(['scil_fodf_ssst.py', in_dwi, in_bval,
                            in_bvec, in_frf, 'fodf.nii.gz', '--sh_order', '4',
                            '--sh_basis', 'tournier07', '--processes', '1'])
    assert ret.success

    # Test wrong b0. Current minimal b-value is 5.
    ret = script_runner.run(['scil_fodf_ssst.py', in_dwi, in_bval,
                            in_bvec, in_frf, 'fodf.nii.gz', '--sh_order', '4',
                            '--sh_basis', 'tournier07', '--processes', '1',
                            '--b0_threshold', '1', '-f'])
    assert not ret.success
