#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()
in_img = os.path.join(SCILPY_HOME, 'others', 'fa.nii.gz')
# fa.nii.gz has a size of 111x133x109


def test_help_option(script_runner):
    ret = script_runner.run('scil_volume_reshape.py', '--help')
    assert ret.success


def test_execution_crop(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_volume_reshape.py', in_img,
                            'fa_reshape.nii.gz', '--volume_size', '90',
                            '-f')
    assert ret.success


def test_execution_pad(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_volume_reshape.py', in_img,
                            'fa_reshape.nii.gz', '--volume_size', '150',
                            '-f')
    assert ret.success


def test_execution_full_size(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_volume_reshape.py', in_img,
                            'fa_reshape.nii.gz', '--volume_size',
                            '164', '164', '164', '-f')
    assert ret.success


def test_execution_dtype(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_volume_reshape.py', in_img,
                            'fa_reshape.nii.gz', '--volume_size',
                            '111', '133', '109', '--data_type',
                            'uint8', '-f')
    assert ret.success


def test_execution_ref(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ref = os.path.join(SCILPY_HOME, 'others', 'fa_resample.nii.gz')
    ret = script_runner.run('scil_volume_reshape.py', in_img,
                            'fa_reshape.nii.gz', '--ref', ref, '-f')
    assert ret.success
