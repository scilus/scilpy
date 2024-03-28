#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()

in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM.trk')

# toDo for more coverage: add a LUT in test data, use option --LUT
# toDo. get a dpp / dps file to load, use options --load_dpp, --load_dps


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_assign_custom_color.py',
                            '--help')
    assert ret.success


def test_execution_from_anat(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_anat = os.path.join(SCILPY_HOME, 'tractometry',
                           'IFGWM_labels_map.nii.gz')

    ret = script_runner.run('scil_tractogram_assign_custom_color.py',
                            in_bundle, 'colored.trk', '--from_anatomy',
                            in_anat, '--out_colorbar', 'test_colorbar.png')
    assert ret.success


def test_execution_along_profile(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_tractogram_assign_custom_color.py',
                            in_bundle, 'colored2.trk', '--along_profile')
    assert ret.success


def test_execution_from_angle(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_tractogram_assign_custom_color.py',
                            in_bundle, 'colored3.trk', '--local_angle')
    assert ret.success
