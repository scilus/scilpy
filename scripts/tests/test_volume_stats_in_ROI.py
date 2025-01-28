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
    ret = script_runner.run('scil_volume_stats_in_ROI.py', '--help')
    assert ret.success


def test_execution_tractometry(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_roi = os.path.join(SCILPY_HOME, 'tractometry',
                          'IFGWM.nii.gz')
    in_ref = os.path.join(SCILPY_HOME, 'tractometry',
                          'mni_masked.nii.gz')

    # Test with a single ROI input
    ret = script_runner.run('scil_volume_stats_in_ROI.py',
                            in_roi, '--metrics', in_ref)
    assert ret.success

    # Test with multiple ROIs input
    ret = script_runner.run('scil_volume_stats_in_ROI.py',
                            in_roi, in_roi, in_roi, '--metrics', in_ref)
    assert ret.success

    # Test with multiple metric input
    ret = script_runner.run('scil_volume_stats_in_ROI.py',
                            in_roi, '--metrics', in_ref, in_ref, in_ref)
    assert ret.success

    # Test with multiple metric and ROIs input
    ret = script_runner.run('scil_volume_stats_in_ROI.py',
                            in_roi, in_roi, '--metrics', in_ref, in_ref)
    assert ret.success

    # Test with a metric folder
    metrics_dir = os.path.join(SCILPY_HOME, 'plot')
    in_roi = os.path.join(SCILPY_HOME, 'plot', 'mask_wm.nii.gz')
    ret = script_runner.run('scil_volume_stats_in_ROI.py',
                            in_roi, '--metrics_dir', metrics_dir)
    assert ret.success
