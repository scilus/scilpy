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
    ret = script_runner.run(['scil_connectivity_structural_graph_measures', '--help'])
    assert ret.success


def test_execution_connectivity(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sc = os.path.join(SCILPY_HOME, 'connectivity', 'sub-001_commit.npy')
    in_len = os.path.join(SCILPY_HOME, 'connectivity', 'sub-001_length.npy')
    ret = script_runner.run(['scil_connectivity_structural_graph_measures',
                             in_sc, 'gtm.json', '--length', in_len,
                             '--avg_node_wise', '--small_world'])
    assert ret.success
