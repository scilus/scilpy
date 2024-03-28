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
    ret = script_runner.run('scil_dwi_powder_average.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')

    ret = script_runner.run('scil_dwi_powder_average.py', in_dwi,
                            in_bval, 'out_pwd_avg.nii.gz', '--shells', '1000')
    assert ret.success
