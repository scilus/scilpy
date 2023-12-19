#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_connectivity_reorder_rois.py', '--help')
    assert ret.success


def test_execution_compute_OLO(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_sc = os.path.join(get_home(), 'connectivity',
                         'sc_norm.npy')
    in_labels_list = os.path.join(get_home(), 'connectivity',
                                  'labels_list.txt')
    ret = script_runner.run('scil_connectivity_reorder_rois.py', in_sc,
                            '--optimal_leaf_ordering', 'OLO.txt',
                            '--out_dir', os.path.expanduser(tmp_dir.name),
                            '--labels_list', in_labels_list, '-f')
    assert ret.success


def test_execution_apply_ordering(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_sc = os.path.join(get_home(), 'connectivity', 'sc_norm.npy')
    in_txt = os.path.join(get_home(), 'connectivity', 'reorder.txt')
    in_labels_list = os.path.join(get_home(), 'connectivity',
                                  'labels_list.txt')
    ret = script_runner.run('scil_connectivity_reorder_rois.py', in_sc,
                            '--in_ordering', in_txt,
                            '--out_suffix', '_sc_reo',
                            '--out_dir', os.path.expanduser(tmp_dir.name),
                            '--labels_list', in_labels_list, '-f')
    assert ret.success
