#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip',
                                           'commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_frf_mean.py', '--help')
    assert ret.success


def test_execution_processing_ssst(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_frf = os.path.join(SCILPY_HOME, 'processing', 'frf.txt')
    ret = script_runner.run('scil_frf_mean.py', in_frf, in_frf, 'mfrf1.txt')
    assert ret.success


def test_execution_processing_msmt(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_frf = os.path.join(SCILPY_HOME, 'commit_amico', 'wm_frf.txt')
    ret = script_runner.run('scil_frf_mean.py', in_frf, in_frf, 'mfrf2.txt')
    assert ret.success


def test_outputs_precision(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_frf = os.path.join(SCILPY_HOME, 'commit_amico', 'wm_frf.txt')
    ret = script_runner.run('scil_frf_mean.py', in_frf, in_frf, 'mfrfp.txt',
                            '--precision', '4')
    assert ret.success

    expected = [
        "0.0016 0.0004 0.0004 3076.7249",
        "0.0012 0.0003 0.0003 3076.7249",
        "0.0009 0.0003 0.0003 3076.7249"
    ]
    with open('mfrfp.txt', 'r') as result:
            for i, line in enumerate(result.readlines()):
                assert line.strip("\n") == expected[i]          


def test_execution_processing_bad_input(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_wm_frf = os.path.join(SCILPY_HOME, 'commit_amico', 'wm_frf.txt')
    in_frf = os.path.join(SCILPY_HOME, 'processing', 'frf.txt')
    ret = script_runner.run('scil_frf_mean.py', in_wm_frf, in_frf, 'mfrf3.txt')
    assert not ret.success
