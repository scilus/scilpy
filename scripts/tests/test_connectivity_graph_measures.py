#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_connectivity_graph_measures.py', '--help')
    assert ret.success


def test_execution_connectivity(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_sc = os.path.join(get_home(), 'connectivity', 'sc_norm.npy')
    in_len = os.path.join(get_home(), 'connectivity', 'len.npy')
    ret = script_runner.run('scil_connectivity_graph_measures.py', in_sc,
                            in_len, 'gtm.json', '--avg_node_wise',
                            '--small_world')
    assert ret.success
