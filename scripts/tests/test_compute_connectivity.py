#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_connectivity.py', '--help')
    assert ret.success


def test_execution_connectivity(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_h5 = os.path.join(get_home(), 'connectivity',
                            'decompose.h5')
    input_atlas = os.path.join(get_home(), 'connectivity',
                               'endpoints_atlas.nii.gz')
    input_avg = os.path.join(get_home(), 'connectivity',
                             'avg_density_maps/')
    input_afd = os.path.join(get_home(), 'connectivity',
                             'afd_max.nii.gz')
    ret = script_runner.run('scil_compute_connectivity.py', input_h5,
                            input_atlas, '--volume', 'vol.npy',
                            '--streamline_count', 'sc.npy',
                            '--length', 'len.npy',
                            '--similarity', input_avg, 'sim.npy',
                            '--metrics', input_afd, 'afd_max.npy',
                            '--density_weighting', '--no_self_connection',
                            '--processes', '1')
    assert ret.success
