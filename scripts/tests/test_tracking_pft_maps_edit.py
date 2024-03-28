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
    ret = script_runner.run('scil_tracking_pft_maps_edit.py', '--help')
    assert ret.success


def test_execution_tracking(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_include = os.path.join(SCILPY_HOME, 'tracking',
                              'map_include.nii.gz')
    in_exclude = os.path.join(SCILPY_HOME, 'tracking',
                              'map_exclude.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking',
                           'seeding_mask.nii.gz')
    ret = script_runner.run('scil_tracking_pft_maps_edit.py',
                            in_include, in_exclude, in_mask,
                            'map_include_corr.nii.gz',
                            'map_exclude_corr.nii.gz')
    assert ret.success
