#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.dvc import pull_test_case_package

# If they already exist, this only takes 5 seconds (check md5sum)
test_data_root = pull_test_case_package("aodf")
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_aodf_metrics.py', '--help')
    assert ret.success


def test_execution(script_runner, monkeypatch):

    # toDo: Add --mask.
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(
        f"{test_data_root}/fodf_descoteaux07_sub_unified_asym.nii.gz")

    # Using a low resolution sphere for peak extraction reduces process time
    ret = script_runner.run('scil_aodf_metrics.py', in_fodf,
                            '--sphere', 'repulsion100')
    assert ret.success


def test_assert_not_all(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(
        f"{test_data_root}/fodf_descoteaux07_sub_unified_asym.nii.gz")

    ret = script_runner.run('scil_aodf_metrics.py', in_fodf,
                            '--not_all')
    assert not ret.success


def test_execution_not_all(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(
        f"{test_data_root}/fodf_descoteaux07_sub_unified_asym.nii.gz")

    ret = script_runner.run('scil_aodf_metrics.py', in_fodf,
                            '--not_all', '--asi_map',
                            'asi_map.nii.gz', '-f')
    assert ret.success


def test_assert_symmetric_input(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(
        f"{test_data_root}/fodf_descoteaux07_sub.nii.gz")

    # Using a low resolution sphere for peak extraction reduces process time
    ret = script_runner.run('scil_aodf_metrics.py', in_fodf,
                            '--sphere', 'repulsion100')
    assert not ret.success


def test_execution_symmetric_input(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(
        f"{test_data_root}/fodf_descoteaux07_sub.nii.gz")

    # Using a low resolution sphere for peak extraction reduces process time
    ret = script_runner.run('scil_aodf_metrics.py', in_fodf,
                            '--sphere', 'repulsion100', '--not_all',
                            '--nufid', 'nufid.nii.gz')
    assert not ret.success
