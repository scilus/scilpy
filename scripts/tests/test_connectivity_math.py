#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_connectivity_math.py', '--help')
    assert ret.success


def test_execution_connectivity_div(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sc = os.path.join(SCILPY_HOME, 'connectivity',
                         'sc.npy')
    in_vol = os.path.join(SCILPY_HOME, 'connectivity',
                          'vol.npy')
    ret = script_runner.run('scil_connectivity_math.py', 'division',
                            in_sc, in_vol, 'sc_norm_vol.npy')
    assert ret.success


def test_execution_connectivity_add(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sc = os.path.join(SCILPY_HOME, 'connectivity',
                         'sc.npy')
    ret = script_runner.run('scil_connectivity_math.py', 'addition',
                            in_sc, '10', 'sc_add_10.npy')
    assert ret.success


def test_execution_connectivity_lower_threshold(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sc = os.path.join(SCILPY_HOME, 'connectivity',
                         'sc.npy')
    ret = script_runner.run('scil_connectivity_math.py', 'lower_threshold',
                            in_sc, '5', 'sc_lower_threshold.npy')
    assert ret.success
