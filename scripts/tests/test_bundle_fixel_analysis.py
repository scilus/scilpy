#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_bundle_fixel_analysis.py', '--help')
    assert ret.success


def test_default_parameters(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_peaks = os.path.join(SCILPY_HOME, 'commit_amico', 'peaks.nii.gz')
    in_bundle = os.path.join(SCILPY_HOME, 'commit_amico', 'tracking.trk')

    # Using multiprocessing in this test, single in following tests.
    ret = script_runner.run('scil_bundle_fixel_analysis.py', in_peaks,
                            '--in_bundles', in_bundle,
                            '--processes', '4', '-f')
    assert ret.success


def test_all_parameters(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_peaks = os.path.join(SCILPY_HOME, 'commit_amico', 'peaks.nii.gz')
    in_bundle = os.path.join(SCILPY_HOME, 'commit_amico', 'tracking.trk')

    ret = script_runner.run('scil_bundle_fixel_analysis.py', in_peaks,
                            '--in_bundles', in_bundle,
                            '--in_bundles_names', 'test',
                            '--abs_thr', '5',
                            '--rel_thr', '0.05',
                            '--norm', 'fixel',
                            '--split_bundles', '--split_fixels',
                            '--single_bundle',
                            '--processes', '1', '-f')
    assert ret.success


def test_multiple_norm(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_peaks = os.path.join(SCILPY_HOME, 'commit_amico', 'peaks.nii.gz')
    in_bundle = os.path.join(SCILPY_HOME, 'commit_amico', 'tracking.trk')

    ret = script_runner.run('scil_bundle_fixel_analysis.py', in_peaks,
                            '--in_bundles', in_bundle,
                            '--in_bundles_names', 'test',
                            '--abs_thr', '5',
                            '--rel_thr', '0.05',
                            '--norm', 'fixel', 'none', 'voxel',
                            '--split_bundles', '--split_fixels',
                            '--single_bundle',
                            '--out_dir', '.',
                            '--processes', '1', '-f')
    assert ret.success
    assert os.path.isfile('bundles_LUT.txt')
    for n in ['voxel', 'fixel', 'none']:
        assert os.path.isfile('fixel_density_maps_{}-norm.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-norm_f1.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-norm_f2.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-norm_f3.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-norm_f4.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-norm_f5.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-'
                              'norm_test.nii.gz'.format(n))
        assert os.path.isfile('nb_bundles_per_fixel_{}-norm.nii.gz'.format(n))
        assert os.path.isfile('nb_bundles_per_voxel_{}-norm.nii.gz'.format(n))
        assert os.path.isfile('single_bundle_mask_{}-'
                              'norm_WM.nii.gz'.format(n))
        assert os.path.isfile('single_bundle_mask_{}-'
                              'norm_test.nii.gz'.format(n))
    
    assert os.path.isfile('voxel_density_maps_voxel-norm.nii.gz')
    assert not os.path.isfile('voxel_density_maps_fixel-norm.nii.gz')
    assert os.path.isfile('voxel_density_maps_none-norm.nii.gz')

# We would need a tractogram with data_per_streamline to test the --dps_key
# option
