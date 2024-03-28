#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()
in_h5 = os.path.join(SCILPY_HOME, 'connectivity', 'decompose.h5')


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_convert_hdf5_to_trk.py', '--help')
    assert ret.success


def test_execution_all_keys(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_tractogram_convert_hdf5_to_trk.py',
                            in_h5, 'save_trk/')
    assert ret.success

    # With current test data, out directory should have 7 files
    out_files = glob.glob('save_trk/*')
    assert len(out_files) == 7


def test_execution_edge_keys(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_tractogram_convert_hdf5_to_trk.py',
                            in_h5, 'save_trk2/', '--edge_keys', '1_10', '1_7')
    assert ret.success

    # Out directory should have 2 files
    out_files = glob.glob('save_trk2/*')
    assert len(out_files) == 2


def test_execution_node_keys(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_tractogram_convert_hdf5_to_trk.py',
                            in_h5, 'save_trk3/', '--node_keys', '7')
    assert ret.success

    # With current test data, out directory should have 3 files
    out_files = glob.glob('save_trk3/*')
    assert len(out_files) == 3


def test_execution_save_empty(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # Saving a txt file with more connections than really exist, to save empty
    # connections.
    with open('labels_list.txt', 'w') as f:
        f.write('1\n10\n100')
    ret = script_runner.run('scil_tractogram_convert_hdf5_to_trk.py',
                            in_h5, 'save_trk4/',
                            '--save_empty', 'labels_list.txt',
                            '--edge_keys', '1_10', '1_100',
                            '-v', 'DEBUG')
    assert ret.success

    # Out directory should have 2 files
    out_files = glob.glob('save_trk4/*')
    assert len(out_files) == 2
