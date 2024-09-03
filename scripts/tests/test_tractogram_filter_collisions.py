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
    ret = script_runner.run('scil_tractogram_filter_collisions.py', '--help')
    assert ret.success


def test_execution_filtering(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = os.path.join(SCILPY_HOME, 'tracking', 'diameters.txt')
    diameters = [0.0012, 0.0024, 0.0018, 0.0043, 0.0017,
                 0.0013, 0.0011, 0.0028, 0.0016, 0.0036] * 100
    np.savetxt(in_diameters, diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            in_tractogram, in_diameters, 'local_clean.trk',
                            '-f')

    assert ret.success


def test_execution_filtering_invalid_collided(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = os.path.join(SCILPY_HOME, 'tracking', 'diameters.txt')
    diameters = [0.0012, 0.0024, 0.0018, 0.0043, 0.0017,
                 0.0013, 0.0011, 0.0028, 0.0016, 0.0036] * 100
    np.savetxt(in_diameters, diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            in_tractogram, in_diameters, 'local_clean.trk',
                            '--save_colliding', '--save_collided', '-f')

    assert ret.success


def test_execution_filtering_single_diameter(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = os.path.join(SCILPY_HOME, 'tracking', 'diameters.txt')
    diameters = [0.0025]
    np.savetxt(in_diameters, diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            in_tractogram, in_diameters, 'local_clean.trk',
                            '--single_diameter', '-f')

    assert ret.success


def test_execution_filtering_shuffle(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = os.path.join(SCILPY_HOME, 'tracking', 'diameters.txt')
    diameters = [0.0012, 0.0024, 0.0018, 0.0043, 0.0017,
                 0.0013, 0.0011, 0.0028, 0.0016, 0.0036] * 100
    np.savetxt(in_diameters, diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            in_tractogram, in_diameters, 'local_clean.trk',
                            '--shuffle', '-f')

    assert ret.success


def test_execution_filtering_min_distance(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = os.path.join(SCILPY_HOME, 'tracking', 'diameters.txt')
    diameters = [0.0012, 0.0024, 0.0018, 0.0043, 0.0017,
                 0.0013, 0.0011, 0.0028, 0.0016, 0.0036] * 100
    np.savetxt(in_diameters, diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            in_tractogram, in_diameters, 'local_clean.trk',
                            '--min_distance', '0.1', '-f')

    assert ret.success


def test_execution_filtering_all_options(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = os.path.join(SCILPY_HOME, 'tracking', 'diameters.txt')
    diameters = [0.0025]
    np.savetxt(in_diameters, diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            in_tractogram, in_diameters, 'local_clean.trk',
                            '--save_colliding', '--save_collided',
                            '--shuffle', '--single_diameter',
                            '--min_distance', '0.1', '-f')

    assert ret.success
