#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tracking_pft_maps.py',
                            '--help')
    assert ret.success


def test_execution_tracking(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_wm = os.path.join(SCILPY_HOME, 'tracking',
                         'map_wm.nii.gz')
    in_gm = os.path.join(SCILPY_HOME, 'tracking',
                         'map_gm.nii.gz')
    in_csf = os.path.join(SCILPY_HOME, 'tracking',
                          'map_csf.nii.gz')
    ret = script_runner.run('scil_tracking_pft_maps.py',
                            in_wm, in_gm, in_csf)
    assert ret.success
