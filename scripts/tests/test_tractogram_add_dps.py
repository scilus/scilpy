#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import tempfile

from dipy.io.streamline import load_tractogram

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_add_dps.py',
                            '--help')
    assert ret.success


def test_execution_add_dps(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    sft = load_tractogram(in_bundle, 'same')
    filename = 'vals.npy'
    outname = 'out.trk'
    np.save(filename, np.arange(len(sft)))
    ret = script_runner.run('scil_tractogram_add_dps.py',
                            in_bundle, filename, 'key', outname, '-f')
    assert ret.success


def test_execution_add_dps_missing_vals(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    sft = load_tractogram(in_bundle, 'same')
    filename = 'vals.npy'
    outname = 'out.trk'
    np.save(filename, np.arange(len(sft) - 10))
    ret = script_runner.run('scil_tractogram_add_dps.py',
                            in_bundle, filename, 'key', outname, '-f')
    assert ret.stderr


def test_execution_add_dps_existing_key(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    sft = load_tractogram(in_bundle, 'same')
    filename = 'vals.npy'
    outname = 'out.trk'
    outname2 = 'out_2.trk'
    np.save(filename, np.arange(len(sft)))
    ret = script_runner.run('scil_tractogram_add_dps.py',
                            in_bundle, filename, 'key', outname, '-f')
    assert ret.success
    ret = script_runner.run('scil_tractogram_add_dps.py',
                            outname, filename, 'key', outname2)
    assert not ret.success


def test_execution_add_dps_tck(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    sft = load_tractogram(in_bundle, 'same')
    filename = 'vals.npy'
    outname = 'out.tck'
    np.save(filename, np.arange(len(sft)))
    ret = script_runner.run('scil_tractogram_add_dps.py',
                            in_bundle, filename, 'key', outname, '-f')
    assert not ret.success
