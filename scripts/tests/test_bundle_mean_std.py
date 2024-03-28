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
    ret = script_runner.run('scil_bundle_mean_std.py', '--help')
    assert ret.success


def test_execution_tractometry_whole(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM.trk')
    in_ref = os.path.join(SCILPY_HOME, 'tractometry', 'mni_masked.nii.gz')
    ret = script_runner.run('scil_bundle_mean_std.py', in_bundle, in_ref,
                            '--density_weighting', '--include_dps')
    assert ret.success


def test_execution_tractometry_per_point(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM.trk')
    in_label = os.path.join(SCILPY_HOME, 'tractometry',
                            'IFGWM_labels_map.nii.gz')
    in_ref = os.path.join(SCILPY_HOME, 'tractometry', 'mni_masked.nii.gz')
    ret = script_runner.run('scil_bundle_mean_std.py', in_bundle, in_ref,
                            '--per_point', in_label, '--density_weighting')

    assert ret.success
