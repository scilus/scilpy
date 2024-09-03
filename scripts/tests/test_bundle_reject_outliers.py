#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_bundle_reject_outliers.py', '--help')
    assert ret.success


def test_execution_filtering(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering', 'bundle_all_1mm.trk')
    ret = script_runner.run('scil_bundle_reject_outliers.py', in_bundle,
                            'inliers.trk', '--alpha', '0.6',
                            '--remaining_bundle', 'outliers.trk',
                            '--display_counts', '--indent', '4',
                            '--sort_keys')
    assert ret.success
