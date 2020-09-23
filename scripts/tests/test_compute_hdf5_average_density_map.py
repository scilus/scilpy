#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_hdf5_average_density_map.py',
                            '--help')
    assert ret.success


def test_execution_connectivity(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_h5 = os.path.join(get_home(), 'connectivity',
                            'decompose.h5')
    ret = script_runner.run('scil_compute_hdf5_average_density_map.py',
                            in_h5, 'avg_density_maps/', '--binary',
                            '--processes', '1')
    assert ret.success
