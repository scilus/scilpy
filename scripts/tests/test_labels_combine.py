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
    ret = script_runner.run('scil_labels_combine.py', '--help')
    assert ret.success


def test_execution_atlas(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_atlas_1 = os.path.join(SCILPY_HOME, 'atlas',
                              'atlas_freesurfer_v2.nii.gz')
    in_brainstem = os.path.join(SCILPY_HOME, 'atlas', 'brainstem.nii.gz')
    ret = script_runner.run('scil_labels_combine.py',
                            'atlas_freesurfer_v2_single_brainstem.nii.gz',
                            '--volume_ids', in_atlas_1, '8', '47', '251',
                            '252', '253', '254', '1022', '1024', '2022',
                            '2024', '--volume_ids', in_brainstem, '16')
    assert ret.success


def test_execution_atlas_merge(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_atlas_1 = os.path.join(SCILPY_HOME, 'atlas',
                              'atlas_freesurfer_v2.nii.gz')
    in_brainstem = os.path.join(SCILPY_HOME, 'atlas', 'brainstem.nii.gz')
    ret = script_runner.run('scil_labels_combine.py',
                            'atlas_freesurfer_v2_merge_brainstem.nii.gz',
                            '--volume_ids', in_atlas_1, '8', '47', '251',
                            '252', '253', '254', '1022', '1024', '2022',
                            '2024', '--volume_ids', in_brainstem, '16',
                            '--merge_groups')
    assert ret.success
