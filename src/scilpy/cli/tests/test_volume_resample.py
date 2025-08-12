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


def test_help_option(script_runner):
    ret = script_runner.run(['scil_volume_resample', '--help'])
    assert ret.success


def test_execution_given_size(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run(['scil_volume_resample', in_img,
                             'fa_resample_2.nii.gz', '--voxel_size', '2'])
    assert ret.success


def test_execution_force_voxel(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run(['scil_volume_resample', in_img,
                             'fa_resample_4.nii.gz', '--voxel_size', '4',
                             '--enforce_voxel_size'])
    assert ret.success


def test_execution_ref(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ref = os.path.join(SCILPY_HOME, 'others', 'fa_resample.nii.gz')
    ret = script_runner.run(['scil_volume_resample', in_img,
                             'fa_resample2.nii.gz', '--ref', ref])
    assert ret.success


def test_execution_ref_force(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ref = os.path.join(SCILPY_HOME, 'others', 'fa_resample.nii.gz')
    ret = script_runner.run(['scil_volume_resample', in_img,
                             'fa_resample_ref.nii.gz', '--ref', ref,
                             '--enforce_dimensions'])
    assert ret.success
