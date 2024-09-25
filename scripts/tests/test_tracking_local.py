#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tracking_local.py', '--help')
    assert ret.success


def test_execution_tracking_fodf_prob(script_runner, monkeypatch):
    # Our tests use -nt 100.
    # Our testing seeding mask has 125 286 voxels, this would be long.
    # Only testing option npv in our first gpu test, below
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'local_prob.trk', '--nt', '100',
                            '--compress', '0.1', '--sh_basis', 'descoteaux07',
                            '--min_length', '20', '--max_length', '200')
    assert ret.success


def test_execution_tracking_fodf_det(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'local_det.trk', '--nt', '100',
                            '--compress', '0.1', '--sh_basis', 'descoteaux07',
                            '--min_length', '20', '--max_length', '200',
                            '--algo', 'det')
    assert ret.success


def test_execution_tracking_ptt(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'local_ptt.trk', '--nt', '100',
                            '--compress', '0.1', '--sh_basis', 'descoteaux07',
                            '--min_length', '20', '--max_length', '200',
                            '--algo', 'ptt')
    assert ret.success


def test_execution_sphere_subdivide(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'local_sphere.trk',
                            '--nt', '100',
                            '--compress', '0.1', '--sh_basis', 'descoteaux07',
                            '--min_length', '20', '--max_length', '200',
                            '--sub_sphere', '2')
    assert ret.success


def test_execution_sphere_gpu(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'sphere_gpu.trk',
                            '--use_gpu', '--sphere', 'symmetric362',
                            '--npv', '1')

    assert not ret.success


def test_sh_interp_without_gpu(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'nearest_interp.trk',
                            '--sh_interp', 'nearest', '--nt', '100')

    assert not ret.success


def test_forward_without_gpu(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'fwd_only.trk',
                            '--forward_only', '--nt', '100')

    assert not ret.success


def test_batch_size_without_gpu(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'batch.trk',
                            '--batch_size', 100)

    assert not ret.success


def test_algo_with_gpu(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'gpu_det.trk', '--algo',
                            'det', '--use_gpu', '--nt', '100')

    assert not ret.success


def test_execution_tracking_fodf_no_compression(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'local_prob2.trk',
                            '--nt', '100', '--sh_basis', 'descoteaux07',
                            '--max_length', '200')

    assert ret.success


def test_execution_tracking_peaks(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_peaks = os.path.join(SCILPY_HOME, 'tracking', 'peaks.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')
    ret = script_runner.run('scil_tracking_local.py', in_peaks,
                            in_mask, in_mask, 'local_eudx.trk', '--nt', '100',
                            '--compress', '0.1', '--sh_basis', 'descoteaux07',
                            '--min_length', '20', '--max_length', '200',
                            '--algo', 'eudx')
    assert ret.success


def test_execution_tracking_fodf_prob_pmf_mapping(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'local_prob3.trk', '--nt', '100',
                            '--compress', '0.1', '--sh_basis', 'descoteaux07',
                            '--min_length', '20', '--max_length', '200',
                            '--sh_to_pmf', '-v')
    assert ret.success


def test_execution_tracking_ptt(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_tracking_local.py', in_fodf,
                            in_mask, in_mask, 'local_ptt.trk', '--nt', '100',
                            '--compress', '0.1', '--sh_basis', 'descoteaux07',
                            '--min_length', '20', '--max_length', '200',
                            '--sh_to_pmf', '-v', '--probe_length', '2',
                            '--probe_quality', '5', '--algo', 'ptt')
    assert ret.success
