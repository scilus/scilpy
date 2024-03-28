#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_bundle_diameter.py', '--help')
    assert ret.success


def test_execution_tractometry(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM.trk')
    in_labels = os.path.join(SCILPY_HOME, 'tractometry',
                             'IFGWM_labels_map.nii.gz')
    ret = script_runner.run('scil_bundle_diameter.py', in_bundle, in_labels,
                            '--wireframe', '--fitting_func', 'lin_up')
    assert ret.success
