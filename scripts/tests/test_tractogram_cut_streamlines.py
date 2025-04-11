#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(),
           keys=['filtering.zip', 'tractograms.zip', 'connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            '--help')
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'filtering',
                                 'bundle_all_1mm.trk')
    in_mask = os.path.join(SCILPY_HOME, 'filtering', 'mask.nii.gz')
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            in_tractogram, 'out_tractogram_cut.trk',
                            '--mask', in_mask, '--min_length', '0', '-f',
                            '--reference', in_mask,
                            '--resample', '0.2', '--compress', '0.1')
    assert ret.success


def test_execution_two_rois(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tractograms',
                                 'streamline_and_mask_operations',
                                 'bundle_4.tck')
    in_mask = os.path.join(SCILPY_HOME, 'tractograms',
                           'streamline_and_mask_operations',
                           'bundle_4_head_tail_offset.nii.gz')
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            in_tractogram, '--mask', in_mask,
                            'out_tractogram_cut.trk', '-f',
                            '--mask', in_mask, '--min_length', '0',
                            '--reference', in_mask,
                            '--resample', '0.2', '--compress', '0.1')
    assert ret.success


def test_execution_keep_longest(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tractograms',
                                 'streamline_and_mask_operations',
                                 'bundle_4.tck')
    in_mask = os.path.join(SCILPY_HOME, 'tractograms',
                           'streamline_and_mask_operations',
                           'bundle_4_head_tail_offset.nii.gz')
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            in_tractogram, '--mask', in_mask,
                            'out_tractogram_cut.trk', '-f',
                            '--keep_longest', '--mask', in_mask,
                            '--min_length', '0', '--resample', '0.2',
                            '--reference', in_mask,
                            '--compress', '0.1')
    assert ret.success


def test_execution_trim_endpoints(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tractograms',
                                 'streamline_and_mask_operations',
                                 'bundle_4.tck')
    in_mask = os.path.join(SCILPY_HOME, 'tractograms',
                           'streamline_and_mask_operations',
                           'bundle_4_head_tail_offset.nii.gz')
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            in_tractogram, '--mask', in_mask,
                            'out_tractogram_cut.trk', '-f',
                            '--trim_endpoints', '--mask', in_mask,
                            '--min_length', '0', '--resample', '0.2',
                            '--reference', in_mask,
                            '--compress', '0.1')
    assert ret.success


def test_execution_labels(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'connectivity',
                                 'bundle_all_1mm.trk')
    in_labels = os.path.join(SCILPY_HOME, 'connectivity',
                             'endpoints_atlas.nii.gz')
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            in_tractogram, '--labels', in_labels,
                            'out_tractogram_cut.trk', '-f',
                            '--label_ids', '1', '10',
                            '--resample', '0.2', '--compress', '0.1')
    assert ret.success


def test_execution_labels_error_trim(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'connectivity',
                                 'bundle_all_1mm.trk')
    in_labels = os.path.join(SCILPY_HOME, 'connectivity',
                             'endpoints_atlas.nii.gz')
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            in_tractogram, '--labels', in_labels,
                            'out_tractogram_cut2.trk', '-f',
                            '--label_ids', '1', '10',
                            '--resample', '0.2', '--compress', '0.1'
                            '--trim_endpoints')
    assert not ret.success


def test_execution_labels_no_point(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'connectivity',
                                 'bundle_all_1mm.trk')
    in_labels = os.path.join(SCILPY_HOME, 'connectivity',
                             'endpoints_atlas.nii.gz')
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            in_tractogram, '--labels', in_labels,
                            'out_tractogram_cut.trk', '-f',
                            '--no_point_in_roi', '--label_ids', '1', '10')
    assert ret.success


def test_execution_labels_one_point(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'connectivity',
                                 'bundle_all_1mm.trk')
    in_labels = os.path.join(SCILPY_HOME, 'connectivity',
                             'endpoints_atlas.nii.gz')
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            in_tractogram, '--labels', in_labels,
                            'out_tractogram_cut.trk', '-f',
                            '--one_point_in_roi', '--label_ids', '1', '10')
    assert ret.success
