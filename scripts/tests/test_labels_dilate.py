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
    ret = script_runner.run('scil_labels_dilate.py', '--help')
    assert ret.success


def test_execution_atlas(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_atlas = os.path.join(SCILPY_HOME, 'atlas',
                            'atlas_freesurfer_v2_single_brainstem.nii.gz')
    ret = script_runner.run('scil_labels_dilate.py', in_atlas,
                            'atlas_freesurfer_v2_single_brainstem_dil.nii.gz',
                            '--processes', '1', '--distance', '2')
    assert ret.success
