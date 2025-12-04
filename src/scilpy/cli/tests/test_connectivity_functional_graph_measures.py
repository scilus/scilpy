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
    ret = script_runner.run(['scil_connectivity_functional_graph_measures', '--help'])
    assert ret.success


def test_execution_connectivity(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_func = os.path.join(SCILPY_HOME, 'connectivity', 'sub-001_functional.npy')
    ret = script_runner.run(['scil_connectivity_functional_graph_measures',
                             in_func, 'gtm.json',
                             '--avg_node_wise'])
    assert ret.success
