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
    ret = script_runner.run(['scil_bundle_compute_endpoints_map', '--help'])
    assert ret.success


def test_execution_tractometry(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM_uni.trk')
    ret = script_runner.run(['scil_bundle_compute_endpoints_map', in_bundle,
                             'head.nii.gz', 'tail.nii.gz', '--binary', '-f'])

    assert ret.success


def test_execution_tractometry_mm_distance5(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM_uni.trk')
    ret = script_runner.run(['scil_bundle_compute_endpoints_map', in_bundle,
                             'head.nii.gz', 'tail.nii.gz', '--binary',
                             '--distance', '5', '--unit', 'mm', '-f'])

    assert ret.success
