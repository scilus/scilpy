#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_connectivity_normalize.py', '--help')
    assert ret.success


def test_execution_connectivity(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sc = os.path.join(SCILPY_HOME, 'connectivity', 'sc.npy')
    in_len = os.path.join(SCILPY_HOME, 'connectivity', 'len.npy')
    in_atlas = os.path.join(SCILPY_HOME, 'connectivity',
                            'endpoints_atlas.nii.gz')
    in_labels_list = os.path.join(SCILPY_HOME, 'connectivity',
                                  'labels_list.txt')
    ret = script_runner.run('scil_connectivity_normalize.py', in_sc,
                            'sc_norm.npy', '--length', in_len,
                            '--parcel_volume', in_atlas, in_labels_list)
    assert ret.success
