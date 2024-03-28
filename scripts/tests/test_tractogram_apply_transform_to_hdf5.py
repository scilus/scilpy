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
    ret = script_runner.run('scil_tractogram_apply_transform_to_hdf5.py',
                            '--help')
    assert ret.success


def test_execution_connectivity(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_h5 = os.path.join(SCILPY_HOME, 'connectivity', 'decompose.h5')
    in_target = os.path.join(SCILPY_HOME, 'connectivity',
                             'endpoints_atlas.nii.gz')
    in_transfo = os.path.join(SCILPY_HOME, 'connectivity', 'affine.txt')

    # toDo. Add a --in_deformation file in our test data, fitting with hdf5.
    #  (See test_tractogram_apply_transform)
    # toDo. Add some dps in the hdf5's data for more line coverage.
    ret = script_runner.run('scil_tractogram_apply_transform_to_hdf5.py',
                            in_h5, in_target, in_transfo, 'decompose_lin.h5')
    assert ret.success
