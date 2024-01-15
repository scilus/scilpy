#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(
        'scil_bundle_mean_fixel_bingham_metric.py', '--help')

    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bingham = os.path.join(get_home(), 'processing', 'fodf_bingham.nii.gz')
    in_metric = os.path.join(get_home(), 'processing', 'fd.nii.gz')
    in_bundles = os.path.join(get_home(), 'processing', 'tracking.trk')

    ret = script_runner.run(
        'scil_bundle_mean_fixel_bingham_metric.py',
        in_bundles, in_bingham, in_metric,
        'fixel_mean_fd.nii.gz', '--length_weighting')

    assert ret.success
