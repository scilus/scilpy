#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bst.zip'])
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_gradients_apply_transform.py', '--help')
    assert ret.success


def test_execution_bst(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bvecs = os.path.join(SCILPY_HOME, 'processing',
                            'dwi.bvec')
    in_aff = os.path.join(SCILPY_HOME, 'bst',
                          'output0GenericAffine.mat')
    ret = script_runner.run('scil_gradients_apply_transform.py',
                            in_bvecs, in_aff,
                            'bvecs_transformed.bvec', '--inverse')
    assert ret.success
