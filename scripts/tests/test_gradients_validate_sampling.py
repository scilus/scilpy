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
    ret = script_runner.run('scil_gradients_validate_sampling.py', '--help')
    assert ret.success


def test_execution_normal(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')

    ret = script_runner.run('scil_gradients_validate_sampling.py', in_bval,
                            in_bvec)
    assert ret.success


def test_execution_mrtrix(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_b = os.path.join(SCILPY_HOME, 'processing',
                           '1000.b')

    ret = script_runner.run('scil_gradients_validate_sampling.py', in_b)
    assert ret.success
