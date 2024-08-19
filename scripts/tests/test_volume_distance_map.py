#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractograms.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_volume_distance_map.py', '--help')
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_mask_1 = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_head_tail.nii.gz')
    in_mask_2 = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_center.nii.gz')
    ret = script_runner.run('scil_volume_distance_map.py',
                            in_mask_1, in_mask_2,
                            'distance_map.nii.gz')

    img = nib.load('distance_map.nii.gz')
    data = img.get_fdata()
    assert data[data > 0].mean() - 17.7777 < 0.0001
    assert ret.success
