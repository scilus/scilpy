#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_denoising_nlmeans.py', '--help')
    assert ret.success


def test_execution_others(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(SCILPY_HOME, 'others', 't1_resample.nii.gz')
    ret = script_runner.run('scil_denoising_nlmeans.py', in_img,
                            't1_denoised.nii.gz',  '4', '--processes', '1')
    assert ret.success
