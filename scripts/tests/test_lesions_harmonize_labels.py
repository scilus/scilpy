#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['lesions.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_lesions_harmonize_labels.py', '--help'])
    assert ret.success

def test_harmonize_label(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    original_data = os.path.join(SCILPY_HOME, 'lesions',
                          '*_lesions_labels.nii.gz')
    ret = script_runner.run(['scil_lesions_harmonize_labels.py',
                            original_data, 'test', '--max_adjacency',
                            '5.0', '--min_voxel_overlap', '1', '-f'])
    assert ret.success