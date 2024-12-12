#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['mrds.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_mrds_select_number_of_tensors.py', '--help')
    assert ret.success


def test_execution_mrds(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_nufo = os.path.join(SCILPY_HOME,
                           'mrds', 'sub-01_nufo.nii.gz')
    # no option
    ret = script_runner.run('scil_mrds_select_number_of_tensors.py',
                            SCILPY_HOME + '/mrds/sub-01',
                            in_nufo,
                            '-f')
    assert ret.success


def test_execution_mrds_w_mask(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_nufo = os.path.join(SCILPY_HOME,
                           'mrds', 'sub-01_nufo.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'mrds',
                           'sub-01_mask.nii.gz')

    ret = script_runner.run('scil_mrds_select_number_of_tensors.py',
                            SCILPY_HOME + '/mrds/sub-01',
                            in_nufo,
                            '--mask', in_mask,
                            '-f')
    assert ret.success
