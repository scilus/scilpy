#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_volume_count_non_zero_voxels.py', '--help')
    assert ret.success


def test_execution_simple_print(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(SCILPY_HOME, 'others', 'rgb.nii.gz')
    ret = script_runner.run('scil_volume_count_non_zero_voxels.py', in_img)
    assert ret.success


def test_execution_save_in_file(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(SCILPY_HOME, 'others', 'rgb.nii.gz')
    ret = script_runner.run('scil_volume_count_non_zero_voxels.py', in_img,
                            '--out', 'printed.txt')
    assert ret.success

    # Then re-use the same out file with --stats
    ret = script_runner.run('scil_volume_count_non_zero_voxels.py', in_img,
                            '--out', 'printed.txt', '--stats')
    assert ret.success
