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
    ret = script_runner.run('scil_ft_fibers_metrics.py', '--help')
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = os.path.join(SCILPY_HOME, 'tracking', 'diameters.txt')
    # Very small diameter to avoid collisions without having to filter
    # This is because this script raises an error when collisions arise
    diameters = [0.0001] * 1000
    np.savetxt(in_diameters, diameters)

    ret = script_runner.run('scil_ft_fibers_metrics.py', in_tractogram,
                            in_diameters, 'metrics.txt', '-f')

    assert ret.success


def test_execution_single_diameter(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = os.path.join(SCILPY_HOME, 'tracking', 'diameters.txt')
    diameters = [0.0001]
    np.savetxt(in_diameters, diameters)

    ret = script_runner.run('scil_ft_fibers_metrics.py', in_tractogram,
                            in_diameters, 'metrics.txt', '--single_diameter',
                            '-f')

    assert ret.success


def test_execution_save_rotation_matrix(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'local.trk')
    in_diameters = os.path.join(SCILPY_HOME, 'tracking', 'diameters.txt')
    diameters = [0.0001] * 1000
    np.savetxt(in_diameters, diameters)

    ret = script_runner.run('scil_ft_fibers_metrics.py', in_tractogram,
                            in_diameters, 'metrics.txt',
                            '--save_rotation_matrix', '-f')

    assert ret.success
