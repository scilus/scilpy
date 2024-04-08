#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_uniformize_endpoints.py',
                            '--help')
    assert ret.success


def test_execution_auto(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM.trk')
    ret = script_runner.run('scil_tractogram_uniformize_endpoints.py',
                            in_bundle, 'IFGWM_uni.trk', '--auto')
    assert ret.success


def test_execution_target_atlas(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM.trk')
    label = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM_labels_map.nii.gz')
    ret = script_runner.run('scil_tractogram_uniformize_endpoints.py',
                            in_bundle, 'IFGWM_uni2.trk', '--target_roi', label,
                            '3', '10')
    assert ret.success
