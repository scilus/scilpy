#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_volume_crop.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing', 'dwi.nii.gz')
    ret = script_runner.run('scil_volume_crop.py', in_dwi, 'dwi_crop.nii.gz',
                            '--output_bbox', 'bbox.pickle')
    assert ret.success

    # Then try to load back the same box
    ret = script_runner.run('scil_volume_crop.py', in_dwi, 'dwi_crop2.nii.gz',
                            '--input_bbox', 'bbox.pickle')
    assert ret.success
