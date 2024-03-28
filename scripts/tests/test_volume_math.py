#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['atlas.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_volume_math.py', '--help')
    assert ret.success


def test_execution_add(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_img_1 = os.path.join(SCILPY_HOME, 'atlas',
                            'brainstem_173.nii.gz')
    in_img_2 = os.path.join(SCILPY_HOME, 'atlas',
                            'brainstem_174.nii.gz')
    in_img_3 = os.path.join(SCILPY_HOME, 'atlas',
                            'brainstem_175.nii.gz')
    ret = script_runner.run('scil_volume_math.py', 'addition',
                            in_img_1, in_img_2, in_img_3, 'brainstem.nii.gz')
    assert ret.success


def test_execution_low_thresh(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(SCILPY_HOME, 'atlas', 'brainstem.nii.gz')
    ret = script_runner.run('scil_volume_math.py', 'lower_threshold',
                            in_img, '1', 'brainstem_bin.nii.gz')
    assert ret.success


def test_execution_low_mult(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(SCILPY_HOME, 'atlas', 'brainstem_bin.nii.gz')
    ret = script_runner.run('scil_volume_math.py', 'multiplication',
                            in_img, '16', 'brainstem_unified.nii.gz')
    assert ret.success


def test_execution_concatenate(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_img_1 = os.path.join(SCILPY_HOME, 'atlas', 'ids', '10.nii.gz')
    in_img_2 = os.path.join(SCILPY_HOME, 'atlas', 'ids', '11.nii.gz')
    in_img_3 = os.path.join(SCILPY_HOME, 'atlas', 'ids', '12.nii.gz')
    in_img_4 = os.path.join(SCILPY_HOME, 'atlas', 'ids', '13.nii.gz')
    in_img_5 = os.path.join(SCILPY_HOME, 'atlas', 'ids', '17.nii.gz')
    in_img_6 = os.path.join(SCILPY_HOME, 'atlas', 'ids', '18.nii.gz')
    ret = script_runner.run('scil_volume_math.py', 'concatenate',
                            in_img_1, in_img_2, in_img_3, in_img_4, in_img_5,
                            in_img_6, 'concat_ids.nii.gz')
    assert ret.success


def test_execution_concatenate_4D(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_img_1 = os.path.join(SCILPY_HOME, 'atlas', 'ids', '10.nii.gz')
    in_img_2 = os.path.join(SCILPY_HOME, 'atlas', 'ids', '8_10.nii.gz')
    in_img_3 = os.path.join(SCILPY_HOME, 'atlas', 'ids', '12.nii.gz')
    in_img_4 = os.path.join(SCILPY_HOME, 'atlas', 'ids', '8_10.nii.gz')
    ret = script_runner.run('scil_volume_math.py', 'concatenate',
                            in_img_1, in_img_2, in_img_3, in_img_4,
                            'concat_ids_4d.nii.gz')
    assert ret.success
