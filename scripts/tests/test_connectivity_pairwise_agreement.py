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
    ret = script_runner.run('scil_connectivity_pairwise_agreement.py',
                            '--help')
    assert ret.success


def test_execution_connectivity(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sc = os.path.join(SCILPY_HOME, 'connectivity', 'sc_norm.npy')
    in_len = os.path.join(SCILPY_HOME, 'connectivity', 'len.npy')
    ret = script_runner.run('scil_connectivity_pairwise_agreement.py', in_sc,
                            in_len, 'diff.json', '--single_compare', in_sc)
    assert ret.success
