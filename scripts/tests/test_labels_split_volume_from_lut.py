#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['atlas.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_labels_split_volume_from_lut.py', '--help')
    assert ret.success


def test_execution_atlas(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_atlas = os.path.join(SCILPY_HOME, 'atlas',
                            'atlas_freesurfer_v2.nii.gz')
    in_json = os.path.join(SCILPY_HOME, 'atlas',
                           'atlas_freesurfer_v2_LUT.json')
    ret = script_runner.run('scil_labels_split_volume_from_lut.py', in_atlas,
                            '--out_prefix', 'brainstem',
                            '--custom_lut', in_json)
    assert ret.success
