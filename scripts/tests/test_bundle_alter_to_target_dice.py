#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_bundle_alter_to_target_dice.py', '--help')
    assert ret.success


def test_execution_subsample(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry',
                             'IFGWM_uni.trk')
    ret = script_runner.run('scil_bundle_alter_to_target_dice.py',
                            in_bundle, 'out_tractogram_subsample.trk',
                            '--min_dice', '0.75', '--epsilon', '0.01',
                            '--subsample', '--shuffle', '-v')
    assert ret.success


def test_execution_trim(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry',
                             'IFGWM_uni.trk')
    ret = script_runner.run('scil_bundle_alter_to_target_dice.py',
                            in_bundle, 'out_tractogram_trim.trk',
                            '--min_dice', '0.75', '--epsilon', '0.01',
                            '--trim', '-v')
    assert ret.success


def test_execution_cut(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry',
                             'IFGWM_uni.trk')
    ret = script_runner.run('scil_bundle_alter_to_target_dice.py',
                            in_bundle, 'out_tractogram_cut.trk',
                            '--min_dice', '0.75', '--epsilon', '0.01',
                            '--cut', '-v', 'DEBUG')
    assert ret.success


def test_execution_replace(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry',
                             'IFGWM_uni.trk')
    ret = script_runner.run('scil_bundle_alter_to_target_dice.py',
                            in_bundle, 'out_tractogram_replace.trk',
                            '--min_dice', '0.75', '--epsilon', '0.01',
                            '--replace', '-v')
    assert ret.success


def test_execution_transform(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry',
                             'IFGWM_uni.trk')
    ret = script_runner.run('scil_bundle_alter_to_target_dice.py',
                            in_bundle, 'out_tractogram_transform.trk',
                            '--min_dice', '0.75', '--epsilon', '0.01',
                            '--transform', '--save_transform',
                            'transform.txt', '-v')
    assert ret.success
    assert os.path.isfile('transform.txt')
