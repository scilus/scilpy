#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip', 'tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_tractogram_compute_density_map',
                            '--help'])
    assert ret.success


def test_execution_others(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'others', 'IFGWM.trk')
    ret = script_runner.run(['scil_tractogram_compute_density_map',
                             in_bundle, 'binary.nii.gz', '--endpoints_only'])
    assert ret.success


def test_execution_tractometry(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM.trk')
    ret = script_runner.run(['scil_tractogram_compute_density_map',
                             in_bundle, 'IFGWM.nii.gz', '--binary'])
    assert ret.success
