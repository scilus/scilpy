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
    ret = script_runner.run('scil_gradients_modify_axes.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # mrtrix
    in_encoding = os.path.join(SCILPY_HOME, 'processing', '1000.b')
    ret = script_runner.run('scil_gradients_modify_axes.py', in_encoding,
                            '1000_flip.b', '-1', '3', '2')
    assert ret.success

    # FSL
    in_encoding = os.path.join(SCILPY_HOME, 'processing', '1000.bvec')
    ret = script_runner.run('scil_gradients_modify_axes.py', in_encoding,
                            '1000_flip.bvec', '1', '-3', '2')
    assert ret.success
