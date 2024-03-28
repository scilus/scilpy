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
    ret = script_runner.run('scil_connectivity_compare_populations.py',
                            '--help')
    assert ret.success


def test_execution_connectivity(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_1 = os.path.join(SCILPY_HOME, 'connectivity', 'sc.npy')
    in_2 = os.path.join(SCILPY_HOME, 'connectivity', 'sc_norm.npy')
    in_mask = os.path.join(SCILPY_HOME, 'connectivity', 'mask.npy')
    ret = script_runner.run('scil_connectivity_compare_populations.py',
                            'pval.npy', '--in_g1', in_1, '--in_g2', in_2,
                            '--filtering_mask', in_mask)
    assert ret.success
