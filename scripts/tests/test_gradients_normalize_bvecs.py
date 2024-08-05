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
    ret = script_runner.run('scil_gradients_normalize_bvecs.py',
                            '--help')
    assert ret.success


def test_execution_processing_fsl(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')
    ret = script_runner.run('scil_gradients_normalize_bvecs.py',
                            in_bvec, '1000_norm.bvec')
    assert ret.success
