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
    ret = script_runner.run('scil_volume_remove_outliers_ransac.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_ad = os.path.join(SCILPY_HOME, 'processing',
                         'ad.nii.gz')
    ret = script_runner.run('scil_volume_remove_outliers_ransac.py', in_ad,
                            'ad_ransanc.nii.gz')
    assert ret.success
