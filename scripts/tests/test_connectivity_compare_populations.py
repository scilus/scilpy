#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_connectivity_compare_populations.py',
                            '--help')
    assert ret.success


def test_execution_connectivity(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_1 = os.path.join(get_home(), 'connectivity', 'sc.npy')
    in_2 = os.path.join(get_home(), 'connectivity', 'sc_norm.npy')
    in_mask = os.path.join(get_home(), 'connectivity', 'mask.npy')
    ret = script_runner.run('scil_connectivity_compare_populations.py',
                            'pval.npy', '--in_g1', in_1, '--in_g2', in_2,
                            '--filtering_mask', in_mask)
    assert ret.success
