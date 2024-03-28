#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['ihMT.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_mti_adjust_B1_header.py', '--help')
    assert ret.success


def test_execution_ihMT_no_option(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_b1_map = os.path.join(SCILPY_HOME,
                             'MT', 'sub-001_run-01_B1map.nii.gz')
    in_b1_json = os.path.join(SCILPY_HOME,
                              'MT', 'sub-001_run-01_B1map.json')

    # no option
    ret = script_runner.run('scil_mti_adjust_B1_header.py', in_b1_map,
                            tmp_dir.name, in_b1_json, '-f')
    assert ret.success
