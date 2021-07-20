#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_apply_transform_to_hdf5.py', '--help')
    assert ret.success


def test_execution_connectivity(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_h5 = os.path.join(get_home(), 'connectivity',
                         'decompose.h5')
    in_target = os.path.join(get_home(), 'connectivity',
                             'endpoints_atlas.nii.gz')
    in_transfo = os.path.join(get_home(), 'connectivity',
                              'affine.txt')
    ret = script_runner.run('scil_apply_transform_to_hdf5.py', in_h5,
                            in_target, in_transfo, 'decompose_lin.h5')
    assert ret.success
