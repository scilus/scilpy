#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(
        'scil_bundle_pairwise_comparison.py', '--help')
    assert ret.success


def test_execution_bundles(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_1 = os.path.join(SCILPY_HOME, 'bundles', 'bundle_0_reco.tck')
    in_2 = os.path.join(SCILPY_HOME, 'bundles', 'voting_results',
                        'bundle_0.trk')
    in_ref = os.path.join(SCILPY_HOME, 'bundles', 'bundle_all_1mm.nii.gz')
    ret = script_runner.run(
        'scil_bundle_pairwise_comparison.py',
        in_1, in_2, 'AF_L_similarity.json',
        '--streamline_dice', '--reference', in_ref,
        '--processes', '1')
    assert ret.success


def test_single(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_1 = os.path.join(SCILPY_HOME, 'bundles', 'bundle_0_reco.tck')
    in_2 = os.path.join(SCILPY_HOME, 'bundles', 'voting_results',
                        'bundle_0.trk')
    in_ref = os.path.join(SCILPY_HOME, 'bundles', 'bundle_all_1mm.nii.gz')
    ret = script_runner.run(
        'scil_bundle_pairwise_comparison.py',
        in_2, 'AF_L_similarity_single.json',
        '--streamline_dice', '--reference', in_ref,
        '--single_compare', in_1,
        '--processes', '1')
    assert ret.success


def test_no_overlap(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_1 = os.path.join(SCILPY_HOME, 'bundles', 'bundle_0_reco.tck')
    in_2 = os.path.join(SCILPY_HOME, 'bundles', 'voting_results',
                        'bundle_0.trk')
    in_ref = os.path.join(SCILPY_HOME, 'bundles', 'bundle_all_1mm.nii.gz')
    ret = script_runner.run(
        'scil_bundle_pairwise_comparison.py', in_1,
        in_2, 'AF_L_similarity_no_overlap.json',
        '--streamline_dice', '--reference', in_ref,
        '--bundle_adjency_no_overlap',
        '--processes', '1')
    assert ret.success


def test_ratio(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_1 = os.path.join(SCILPY_HOME, 'bundles', 'bundle_0_reco.tck')
    in_2 = os.path.join(SCILPY_HOME, 'bundles', 'voting_results',
                        'bundle_0.trk')
    in_ref = os.path.join(SCILPY_HOME, 'bundles', 'bundle_all_1mm.nii.gz')
    ret = script_runner.run(
        'scil_bundle_pairwise_comparison.py',
        in_2, 'AF_L_similarity_ratio.json',
        '--streamline_dice', '--reference', in_ref,
        '--single_compare', in_1,
        '--processes', '1',
        '--ratio')
    assert ret.success


def test_ratio_fail(script_runner, monkeypatch):
    """ Test ratio without single_compare argument.
    The test should fail.
    """
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_1 = os.path.join(SCILPY_HOME, 'bundles',  'bundle_0_reco.tck')
    in_2 = os.path.join(SCILPY_HOME, 'bundles', 'voting_results',
                        'bundle_0.trk')
    in_ref = os.path.join(SCILPY_HOME, 'bundles', 'bundle_all_1mm.nii.gz')
    ret = script_runner.run(
        'scil_bundle_pairwise_comparison.py',
        in_1, in_2, 'AF_L_similarity_fail.json',
        '--streamline_dice', '--reference', in_ref,
        '--processes', '1',
        '--ratio')
    assert not ret.success
