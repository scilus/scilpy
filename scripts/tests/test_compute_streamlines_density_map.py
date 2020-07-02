#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip', 'tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_streamlines_density_map.py',
                            '--help')
    assert ret.success


def test_execution_others(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'others',
                             'IFGWM.trk')
    ret = script_runner.run('scil_compute_streamlines_density_map.py',
                            in_bundle, 'binary.nii.gz', '--binary')
    assert ret.success


def test_execution_tractometry(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'tractometry',
                             'IFGWM.trk')
    ret = script_runner.run('scil_compute_streamlines_density_map.py',
                            in_bundle, 'IFGWM.nii.gz', '--binary')
    assert ret.success
