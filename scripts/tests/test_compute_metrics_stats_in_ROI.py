#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_metrics_stats_in_ROI.py', '--help')
    assert ret.success


def test_execution_tractometry(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_mask = os.path.join(get_home(), 'tractometry',
                           'IFGWM.nii.gz')
    in_ref = os.path.join(get_home(), 'tractometry',
                          'mni_masked.nii.gz')
    ret = script_runner.run('scil_compute_metrics_stats_in_ROI.py',
                            in_mask, '--metrics', in_ref)
    assert ret.success
