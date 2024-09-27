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
    ret = script_runner.run('scil_connectivity_compute_matrices.py', '--help')
    assert ret.success


def test_execution_connectivity(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_h5 = os.path.join(SCILPY_HOME, 'connectivity', 'decompose.h5')
    in_atlas = os.path.join(SCILPY_HOME, 'connectivity',
                            'endpoints_atlas.nii.gz')
    in_avg = os.path.join(SCILPY_HOME, 'connectivity', 'avg_density_maps/')
    in_afd = os.path.join(SCILPY_HOME, 'connectivity', 'afd_max.nii.gz')
    ret = script_runner.run('scil_connectivity_compute_matrices.py', in_h5,
                            in_atlas, '--volume', 'vol.npy',
                            '--streamline_count', 'sc.npy',
                            '--length', 'len.npy',
                            '--similarity', in_avg, 'sim.npy',
                            '--metrics', in_afd, 'afd_max.npy',
                            '--density_weighting', '--no_self_connection',
                            '--processes', '1')
    assert ret.success
