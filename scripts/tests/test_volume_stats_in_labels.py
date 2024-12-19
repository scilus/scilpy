#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['plot.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_volume_stats_in_labels.py', '--help')
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_metric = os.path.join(SCILPY_HOME, 'plot', 'fa.nii.gz')
    in_atlas = os.path.join(SCILPY_HOME, 'plot', 'atlas_brainnetome.nii.gz')
    atlas_lut = os.path.join(SCILPY_HOME, 'plot', 'atlas_brainnetome.json')

    # Test with a single metric
    ret = script_runner.run('scil_volume_stats_in_labels.py',
                            in_atlas, atlas_lut, "--metrics", in_metric)
    assert ret.success

    # Test with multiple metrics
    ret = script_runner.run('scil_volume_stats_in_labels.py',
                            in_atlas, atlas_lut, "--metrics",
                            in_metric, in_metric, in_metric)
    assert ret.success

    # Test with a metric folder
    metrics_dir = os.path.join(SCILPY_HOME, 'plot')
    ret = script_runner.run('scil_volume_stats_in_labels.py',
                            in_atlas, atlas_lut, "--metrics_dir",
                            metrics_dir)
    assert ret.success
