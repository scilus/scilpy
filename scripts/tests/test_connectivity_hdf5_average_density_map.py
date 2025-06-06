#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_connectivity_hdf5_average_density_map.py',
                            '--help')
    assert ret.success


def test_execution_connectivity(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_h5 = os.path.join(SCILPY_HOME, 'connectivity', 'decompose.h5')
    ret = script_runner.run('scil_connectivity_hdf5_average_density_map.py',
                            in_h5, 'avg_density_map/', '--binary',
                            '--processes', '1')
    assert ret.success


def test_execution_connectivity_(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_h5_1 = os.path.join(SCILPY_HOME, 'connectivity', 'decompose.h5')
    in_h5_2 = os.path.join(SCILPY_HOME, 'connectivity', 'decompose_afd_rd.h5')
    ret = script_runner.run('scil_connectivity_hdf5_average_density_map.py',
                            in_h5_1, in_h5_2, 'avg_density_maps/',
                            '--processes', '1')
    assert ret.success
