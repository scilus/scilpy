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
    ret = script_runner.run('scil_gradients_validate_correct_eddy.py',
                            '--help')
    assert ret.success


def test_execution_extract_half(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bvec = os.path.join(SCILPY_HOME, 'processing', 'dwi.bvec')
    in_bval = os.path.join(SCILPY_HOME, 'processing', 'dwi.bval')
    ret = script_runner.run('scil_gradients_validate_correct_eddy.py',
                            in_bvec, in_bval, "32",
                            'out.bvec',
                            'out.bval',
                            '-f')
    assert ret.success

def test_execution_extract_total(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bvec = os.path.join(SCILPY_HOME, 'processing', 'dwi.bvec')
    in_bval = os.path.join(SCILPY_HOME, 'processing', 'dwi.bval')
    ret = script_runner.run('scil_gradients_validate_correct_eddy.py',
                            in_bvec, in_bval, "64",
                            'out.bvec',
                            'out.bval',
                            '-f')
    assert ret.success
