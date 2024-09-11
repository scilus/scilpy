#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractograms.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_fibertube_tracking.py', '--help')
    assert ret.success


def test_execution_(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = 'diameters.txt'
    diameters = [0.0012, 0.0024, 0.0018, 0.0043, 0.0017,
                 0.0013, 0.0011, 0.0028, 0.0016, 0.0036] * 100
    np.savetxt(in_diameters, diameters)
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_fibertube_tracking.py',
                            in_tractogram, in_diameters, in_mask,
                            'tracking.trk', '1', '1', '--nb_seeds_per_fiber',
                            '1', '--nb_fibers', '1', '-f')

    assert ret.success


def test_execution_single_diameter(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = 'diameters.txt'
    diameters = [0.0025]
    np.savetxt(in_diameters, diameters)
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_fibertube_tracking.py',
                            in_tractogram, in_diameters, in_mask,
                            'tracking.trk', '1', '1', '--nb_seeds_per_fiber',
                            '1', '--nb_fibers', '1', '--single_diameter', '-f')

    assert ret.success


def test_execution_forward_only(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = 'diameters.txt'
    diameters = [0.0012, 0.0024, 0.0018, 0.0043, 0.0017,
                 0.0013, 0.0011, 0.0028, 0.0016, 0.0036] * 100
    np.savetxt(in_diameters, diameters)
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_fibertube_tracking.py',
                            in_tractogram, in_diameters, in_mask,
                            'tracking.trk', '1', '1', '--nb_seeds_per_fiber',
                            '1', '--nb_fibers', '1', '--forward_only', '-f')

    assert ret.success


def test_execution_no_compression(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = 'diameters.txt'
    diameters = [0.0012, 0.0024, 0.0018, 0.0043, 0.0017,
                 0.0013, 0.0011, 0.0028, 0.0016, 0.0036] * 100
    np.savetxt(in_diameters, diameters)
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_fibertube_tracking.py',
                            in_tractogram, in_diameters, in_mask,
                            'tracking.trk', '1', '1', '--nb_seeds_per_fiber',
                            '1', '--nb_fibers', '1', '--do_not_compress', '-f')

    assert ret.success


def test_execution_saving(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = 'diameters.txt'
    diameters = [0.0012, 0.0024, 0.0018, 0.0043, 0.0017,
                 0.0013, 0.0011, 0.0028, 0.0016, 0.0036] * 100
    np.savetxt(in_diameters, diameters)
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_fibertube_tracking.py',
                            in_tractogram, in_diameters, in_mask,
                            'tracking.trk', '1', '1', '--nb_seeds_per_fiber',
                            '1', '--nb_fibers', '1', '--save_seeds',
                            '--save_config', '-f')

    assert ret.success


def test_execution_shuffle(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = 'diameters.txt'
    diameters = [0.0012, 0.0024, 0.0018, 0.0043, 0.0017,
                 0.0013, 0.0011, 0.0028, 0.0016, 0.0036] * 100
    np.savetxt(in_diameters, diameters)
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_fibertube_tracking.py',
                            in_tractogram, in_diameters, in_mask,
                            'tracking.trk', '1', '1', '--nb_seeds_per_fiber',
                            '1', '--nb_fibers', '1', '--shuffle', '-f')

    assert ret.success
