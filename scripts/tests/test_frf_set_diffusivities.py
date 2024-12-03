#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(),
           keys=['processing.zip', 'commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_frf_set_diffusivities.py', '--help')
    assert ret.success


def test_execution_processing_ssst(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_frf = os.path.join(SCILPY_HOME, 'processing', 'frf.txt')
    ret = script_runner.run('scil_frf_set_diffusivities.py', in_frf,
                            '15,4,4', 'new_frf.txt', '-f')
    assert ret.success


def test_execution_processing_msmt(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_frf = os.path.join(SCILPY_HOME, 'commit_amico', 'wm_frf.txt')
    ret = script_runner.run('scil_frf_set_diffusivities.py', in_frf,
                            '15,4,4,13,4,4,12,5,5', 'new_frf.txt', '-f')
    assert ret.success


def test_outputs_precision(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_frf = os.path.join(SCILPY_HOME, 'commit_amico', 'wm_frf.txt')
    ret = script_runner.run('scil_frf_set_diffusivities.py', in_frf,
                            '15,4,4,13,4,4,12,5,5', 'new_frf.txt',
                            '--precision', '4', '-f')
    assert ret.success

    expected = [
        "0.0015 0.0004 0.0004 3076.7249",
        "0.0013 0.0004 0.0004 3076.7249",
        "0.0012 0.0005 0.0005 3076.7249"
    ]
    with open('new_frf.txt', 'r') as result:
        for i, line in enumerate(result.readlines()):
            assert line.strip("\n") == expected[i] 


def test_execution_processing__wrong_input(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_frf = os.path.join(SCILPY_HOME, 'commit_amico', 'wm_frf.txt')
    ret = script_runner.run('scil_frf_set_diffusivities.py', in_frf,
                            '15,4,4,13,4,4', 'new_frf.txt', '-f')
    assert not ret.success
