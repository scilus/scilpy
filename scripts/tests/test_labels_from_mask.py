#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractograms.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_labels_from_mask.py', '--help')
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_mask = os.path.join(SCILPY_HOME, 'tractograms',
                           'streamline_and_mask_operations',
                           'bundle_4_head_tail_offset.nii.gz')
    ret = script_runner.run('scil_labels_from_mask.py',
                            in_mask, 'labels_from_mask.nii.gz',
                            '--min_volume', '0', '-f')
    assert ret.success


def test_execution_labels(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_mask = os.path.join(SCILPY_HOME, 'tractograms',
                           'streamline_and_mask_operations',
                           'bundle_4_head_tail_offset.nii.gz')
    ret = script_runner.run('scil_labels_from_mask.py',
                            in_mask, 'labels_from_mask.nii.gz',
                            '--labels', '4', '6', '-f')
    assert ret.success


def test_execution_background(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_mask = os.path.join(SCILPY_HOME, 'tractograms',
                           'streamline_and_mask_operations',
                           'bundle_4_head_tail_offset.nii.gz')
    ret = script_runner.run('scil_labels_from_mask.py',
                            in_mask, 'labels_from_mask.nii.gz',
                            '--background_label', '9', '-f')
    assert ret.success


def test_execution_error(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_mask = os.path.join(SCILPY_HOME, 'tractograms',
                           'streamline_and_mask_operations',
                           'bundle_4_head_tail_offset.nii.gz')
    ret = script_runner.run('scil_labels_from_mask.py',
                            in_mask, 'labels_from_mask.nii.gz',
                            '--labels', '1')
    assert not ret.success


def test_execution_warning(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_mask = os.path.join(SCILPY_HOME, 'tractograms',
                           'streamline_and_mask_operations',
                           'bundle_4_head_tail_offset.nii.gz')
    ret = script_runner.run('scil_labels_from_mask.py',
                            in_mask, 'labels_from_mask.nii.gz',
                            '--labels', '1', '2', '3', '-f')
    assert ret.success
    assert ret.stderr  # Check if there is a warning message


def test_execution_background_warning(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_mask = os.path.join(SCILPY_HOME, 'tractograms',
                           'streamline_and_mask_operations',
                           'bundle_4_head_tail_offset.nii.gz')
    ret = script_runner.run('scil_labels_from_mask.py',
                            in_mask, 'labels_from_mask.nii.gz',
                            '--background_label', '1', '-f')
    assert ret.success
    assert ret.stderr  # Check if there is a warning message
