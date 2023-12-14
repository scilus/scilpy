#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tracking_pft_maps_edit.py', '--help')
    assert ret.success


def test_execution_tracking(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_include = os.path.join(get_home(), 'tracking',
                              'map_include.nii.gz')
    in_exclude = os.path.join(get_home(), 'tracking',
                              'map_exclude.nii.gz')
    in_mask = os.path.join(get_home(), 'tracking',
                           'seeding_mask.nii.gz')
    ret = script_runner.run('scil_tracking_pft_maps_edit.py',
                            in_include, in_exclude, in_mask,
                            'map_include_corr.nii.gz',
                            'map_exclude_corr.nii.gz')
    assert ret.success
