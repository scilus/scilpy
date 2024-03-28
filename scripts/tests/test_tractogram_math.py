#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()
trk_path = os.path.join(SCILPY_HOME, 'others')


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_math.py', '--help')
    assert ret.success


def test_execution_lazy_concatenate_no_color(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'lazy_concatenate',
                            in_tracto_1, in_tracto_2,
                            'lazy_concatenate.trk')
    assert ret.success


def test_execution_lazy_concatenate_mix(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles_color.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'lazy_concatenate',
                            in_tracto_1, in_tracto_2,
                            'lazy_concatenate_mix.trk')
    assert ret.success


def test_execution_union_no_color(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'union',
                            in_tracto_1, in_tracto_2, 'union.trk')
    assert ret.success


def test_execution_intersection_no_color(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'intersection',
                            in_tracto_1, in_tracto_2, 'intersection.trk')
    assert ret.success


def test_execution_difference_no_color(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'difference',
                            in_tracto_1, in_tracto_2, 'difference.trk')
    assert ret.success


def test_execution_concatenate_no_color(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'concatenate',
                            in_tracto_1, in_tracto_2, 'concatenate.trk')
    assert ret.success


def test_execution_union_no_color_robust(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'union',
                            in_tracto_1, in_tracto_2, 'union_r.trk',
                            '--robust')
    assert ret.success


def test_execution_intersection_no_color_robust(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'intersection',
                            in_tracto_1, in_tracto_2, 'intersection_r.trk',
                            '--robust')
    assert ret.success


def test_execution_difference_no_color_robust(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'difference',
                            in_tracto_1, in_tracto_2, 'difference_r.trk',
                            '--robust')
    assert ret.success


def test_execution_union_color(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles_color.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0_color.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'union',
                            in_tracto_1, in_tracto_2, 'union_color.trk')
    assert ret.success


def test_execution_intersection_color(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles_color.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0_color.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'intersection',
                            in_tracto_1, in_tracto_2, 'intersection_color.trk')
    assert ret.success


def test_execution_difference_color(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles_color.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0_color.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'difference',
                            in_tracto_1, in_tracto_2, 'difference_color.trk')
    assert ret.success


def test_execution_concatenate_color(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles_color.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0_color.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'concatenate',
                            in_tracto_1, in_tracto_2, 'concatenate_color.trk')
    assert ret.success


def test_execution_union_mix(script_runner, monkeypatch):
    # This is intentionally failing
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles_color.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'union',
                            in_tracto_1, in_tracto_2, 'union_mix.trk')
    assert not ret.success


def test_execution_intersection_mix_fake(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundles_color.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'intersection',
                            in_tracto_1, in_tracto_2, 'intersection_mix.trk',
                            '--fake_metadata')
    assert ret.success


def test_execution_difference_empty_result(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundle_0.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0_color.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'difference',
                            in_tracto_1, in_tracto_2,
                            'difference_empty_results.trk', '--no_metadata')
    assert ret.success


def test_execution_difference_empty_input_1(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'empty.trk')
    in_tracto_2 = os.path.join(trk_path, 'fibercup_bundle_0_color.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'difference',
                            in_tracto_1, in_tracto_2, 'difference_empty_1.trk',
                            '--no_metadata')
    assert ret.success


def test_execution_difference_empty_input_2(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(trk_path, 'fibercup_bundle_0_color.trk')
    in_tracto_2 = os.path.join(trk_path, 'empty.trk')
    ret = script_runner.run('scil_tractogram_math.py', 'difference',
                            in_tracto_1, in_tracto_2, 'difference_empty_2.trk',
                            '--no_metadata')
    assert ret.success
