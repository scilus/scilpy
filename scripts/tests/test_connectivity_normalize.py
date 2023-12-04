#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_connectivity_normalize.py', '--help')
    assert ret.success


def test_execution_connectivity(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_sc = os.path.join(get_home(), 'connectivity', 'sc.npy')
    in_len = os.path.join(get_home(), 'connectivity', 'len.npy')
    in_atlas = os.path.join(get_home(), 'connectivity',
                            'endpoints_atlas.nii.gz')
    in_labels_list = os.path.join(get_home(), 'connectivity',
                                  'labels_list.txt')
    ret = script_runner.run('scil_connectivity_normalize.py', in_sc,
                            'sc_norm.npy', '--length', in_len,
                            '--parcel_volume', in_atlas, in_labels_list)
    assert ret.success
