#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import tempfile

from dipy.io.streamline import load_tractogram, save_tractogram

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            '--help')
    assert ret.success


def test_execution_dps_math_import(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    sft = load_tractogram(in_bundle, 'same')
    filename = 'vals.npy'
    outname = 'out.trk'
    np.save(filename, np.arange(len(sft)))
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            in_bundle, 'import', 'key',
                            '--in_dps_file', filename,
                            '--out_tractogram', outname,
                            '-f')
    assert ret.success


def test_execution_dps_math_import_single_value(script_runner,
                                                monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    outname = 'out.trk'
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            in_bundle, 'import', 'key',
                            '--in_dps_single_value', '42',
                            '--out_tractogram', outname,
                            '-f')
    assert ret.success


def test_execution_dps_math_import_single_value_array(script_runner,
                                                      monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    outname = 'out.trk'
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            in_bundle, 'import', 'key',
                            '--in_dps_single_value', '1', '1.1', '1.2',
                            '--out_tractogram', outname,
                            '-f')
    assert ret.success


def test_execution_dps_math_import_with_missing_vals(script_runner,
                                                     monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    sft = load_tractogram(in_bundle, 'same')
    filename = 'vals.npy'
    outname = 'out.trk'
    np.save(filename, np.arange(len(sft) - 10))
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            in_bundle, 'import', 'key',
                            '--in_dps_file', filename,
                            '--out_tractogram', outname,
                            '-f')
    assert ret.stderr


def test_execution_dps_math_import_with_existing_key(script_runner,
                                                     monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    sft = load_tractogram(in_bundle, 'same')
    filename = 'vals.npy'
    outname = 'out.trk'
    outname2 = 'out_2.trk'
    np.save(filename, np.arange(len(sft)))
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            in_bundle, 'import', 'key',
                            '--in_dps_file', filename,
                            '--out_tractogram', outname,
                            '-f')
    assert ret.success
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            outname, 'import', 'key',
                            '--in_dps_file', filename,
                            '--out_tractogram', outname2,)
    assert not ret.success


def test_execution_dps_math_tck_output(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    sft = load_tractogram(in_bundle, 'same')
    filename = 'vals.npy'
    outname = 'out.tck'
    np.save(filename, np.arange(len(sft)))
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            in_bundle, 'import', 'key',
                            '--in_dps_file', filename,
                            '--out_tractogram', outname,
                            '-f')
    assert not ret.success


def test_execution_dps_math_delete(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle_no_key = os.path.join(SCILPY_HOME, 'filtering',
                                    'bundle_4.trk')
    in_bundle = 'bundle_4.trk'
    sft = load_tractogram(in_bundle_no_key, 'same')
    sft.data_per_streamline = {
        "key": [0] * len(sft)
    }
    save_tractogram(sft, in_bundle)
    outname = 'out.trk'
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            in_bundle, 'delete', 'key',
                            '--out_tractogram', outname,
                            '-f')
    assert ret.success


def test_execution_dps_math_delete_no_key(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    outname = 'out.trk'
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            in_bundle, 'delete', 'key',
                            '--out_tractogram', outname,
                            '-f')
    assert not ret.success


def test_execution_dps_math_export(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle_no_key = os.path.join(SCILPY_HOME, 'filtering',
                                    'bundle_4.trk')
    in_bundle = 'bundle_4.trk'
    sft = load_tractogram(in_bundle_no_key, 'same')
    sft.data_per_streamline = {
        "key": [0] * len(sft)
    }
    save_tractogram(sft, in_bundle)
    filename = 'out.txt'
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            in_bundle, 'export', 'key',
                            '--out_dps_file', filename,
                            '-f')
    assert ret.success


def test_execution_dps_math_export_no_key(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    filename = 'out.txt'
    ret = script_runner.run('scil_tractogram_dps_math.py',
                            in_bundle, 'export', 'key',
                            '--out_dps_file', filename,
                            '-f')
    assert not ret.success
