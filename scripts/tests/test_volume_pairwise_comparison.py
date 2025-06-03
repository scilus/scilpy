#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import get_testing_files_dict, fetch_data


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=[
           'atlas.zip', 'tractograms.zip', 'tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_volume_pairwise_comparison.py', '--help')
    assert ret.success


def test_label_comparison(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_atlas = os.path.join(SCILPY_HOME, 'atlas',
                            'atlas_freesurfer_v2_single_brainstem.nii.gz')
    in_atlas_dilated = os.path.join(
        SCILPY_HOME, 'atlas',
        'atlas_freesurfer_v2_single_brainstem_dil.nii.gz')
    ret = script_runner.run('scil_volume_pairwise_comparison.py',
                            in_atlas, in_atlas_dilated, 'atlas.json')
    assert ret.success


def test_binary_comparison(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bin_1 = os.path.join(SCILPY_HOME, 'tractograms',
                            'streamline_and_mask_operations',
                            'bundle_4_head_tail.nii.gz')
    in_bin_2 = os.path.join(SCILPY_HOME, 'tractograms',
                            'streamline_and_mask_operations',
                            'bundle_4_head_tail_offset.nii.gz')

    ret = script_runner.run('scil_volume_pairwise_comparison.py',
                            in_bin_1, in_bin_2, 'binary.json')
    assert ret.success


def test_multiple_compare(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bin_1 = os.path.join(SCILPY_HOME, 'tractograms',
                            'streamline_and_mask_operations',
                            'bundle_4_head_tail.nii.gz')
    in_bin_2 = os.path.join(SCILPY_HOME, 'tractograms',
                            'streamline_and_mask_operations',
                            'bundle_4_head_tail_offset.nii.gz')

    in_bin_3 = os.path.join(SCILPY_HOME, 'tractograms',
                            'streamline_and_mask_operations',
                            'bundle_4_center.nii.gz')

    ret = script_runner.run('scil_volume_pairwise_comparison.py',
                            in_bin_1, in_bin_2, in_bin_3, 'multiple.json')
    assert ret.success


def test_single_compare(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bin_1 = os.path.join(SCILPY_HOME, 'tractograms',
                            'streamline_and_mask_operations',
                            'bundle_4_head_tail.nii.gz')
    in_bin_2 = os.path.join(SCILPY_HOME, 'tractograms',
                            'streamline_and_mask_operations',
                            'bundle_4_head_tail_offset.nii.gz')

    in_bin_3 = os.path.join(SCILPY_HOME, 'tractograms',
                            'streamline_and_mask_operations',
                            'bundle_4_center.nii.gz')

    ret = script_runner.run('scil_volume_pairwise_comparison.py',
                            in_bin_1, in_bin_2, 'single.json',
                            '--single_compare', in_bin_3)
    assert ret.success


def test_ratio_compare(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bin_1 = os.path.join(SCILPY_HOME, 'tractograms',
                            'streamline_and_mask_operations',
                            'bundle_4_head_tail.nii.gz')
    in_bin_2 = os.path.join(SCILPY_HOME, 'tractograms',
                            'streamline_and_mask_operations',
                            'bundle_4_head_tail_offset.nii.gz')

    in_bin_3 = os.path.join(SCILPY_HOME, 'tractograms',
                            'streamline_and_mask_operations',
                            'bundle_4_center.nii.gz')

    ret = script_runner.run('scil_volume_pairwise_comparison.py',
                            in_bin_1, in_bin_2, 'ratio.json',
                            '--single_compare', in_bin_3,
                            '--ratio')
    assert ret.success


def test_labels_to_mask_compare(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_atlas = os.path.join(SCILPY_HOME, 'atlas',
                            'atlas_freesurfer_v2_no_brainstem.nii.gz')
    in_mask = os.path.join(
        SCILPY_HOME, 'atlas',
        'brainstem_bin.nii.gz')
    ret = script_runner.run('scil_volume_pairwise_comparison.py',
                            in_atlas, 'labels_to_maskjson',
                            '--single_compare', in_mask,
                            '--labels_to_mask')
    assert ret.success
