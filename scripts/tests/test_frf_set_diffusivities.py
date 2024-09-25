#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

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


def test_execution_processing__wrong_input(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_frf = os.path.join(SCILPY_HOME, 'commit_amico', 'wm_frf.txt')
    ret = script_runner.run('scil_frf_set_diffusivities.py', in_frf,
                            '15,4,4,13,4,4', 'new_frf.txt', '-f')
    assert not ret.success
