#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_gradients_convert.py',
                            '--help')
    assert ret.success


def test_execution_processing_fsl(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')
    ret = script_runner.run('scil_gradients_convert.py',
                            '--input_fsl',
                            in_bval, in_bvec, '1000')
    assert ret.success


def test_execution_processing_mrtrix(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_encoding = os.path.join(SCILPY_HOME, 'processing',
                               '1000.b')
    ret = script_runner.run('scil_gradients_convert.py',
                            '--input_mrtrix',
                            in_encoding, '1000')
    assert ret.success


def test_name_validation_mrtrix(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')
    ret = script_runner.run('scil_gradients_convert.py',
                            '--input_fsl',
                            in_bval, in_bvec, '1000_test.b')
    assert ret.success

    wrong_path = os.path.join(tmp_dir.name, '1000_test.b.b')
    assert not os.path.isfile(wrong_path)

    right_path = os.path.join(tmp_dir.name, '1000_test.b')
    assert os.path.isfile(right_path)


def test_name_validation_fsl_bval(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_encoding = os.path.join(SCILPY_HOME, 'processing',
                               '1000.b')
    ret = script_runner.run('scil_gradients_convert.py',
                            '--input_mrtrix',
                            in_encoding, '1000_test.bval')
    assert ret.success

    wrong_path_bval = os.path.join(tmp_dir.name, '1000_test.bval.bval')
    assert not os.path.isfile(wrong_path_bval)
    wrong_path_bvec = os.path.join(tmp_dir.name, '1000_test.bval.bvec')
    assert not os.path.isfile(wrong_path_bvec)

    right_path_bval = os.path.join(tmp_dir.name, '1000_test.bval')
    assert os.path.isfile(right_path_bval)
    right_path_bvec = os.path.join(tmp_dir.name, '1000_test.bvec')
    assert os.path.isfile(right_path_bvec)


def test_name_validation_fsl_bvec(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_encoding = os.path.join(SCILPY_HOME, 'processing',
                               '1000.b')
    ret = script_runner.run('scil_gradients_convert.py',
                            '--input_mrtrix',
                            in_encoding, '1000_test.bvec')
    assert ret.success

    wrong_path_bval = os.path.join(tmp_dir.name, '1000_test.bvec.bval')
    assert not os.path.isfile(wrong_path_bval)
    wrong_path_bvec = os.path.join(tmp_dir.name, '1000_test.bvec.bvec')
    assert not os.path.isfile(wrong_path_bvec)

    right_path_bval = os.path.join(tmp_dir.name, '1000_test.bval')
    assert os.path.isfile(right_path_bval)
    right_path_bvec = os.path.join(tmp_dir.name, '1000_test.bvec')
    assert os.path.isfile(right_path_bvec)
