#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            '--help')
    assert ret.success


def test_execution_two_roi(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'filtering',
                                 'bundle_all_1mm.trk')
    in_mask = os.path.join(SCILPY_HOME, 'filtering', 'mask.nii.gz')
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            in_tractogram, 'out_tractogram_cut.trk',
                            '--mask', in_mask,
                            '--resample', '0.2', '--compress', '0.1')
    assert ret.success


def test_execution_biggest(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'filtering',
                                 'bundle_all_1mm.trk')
    in_mask = os.path.join(SCILPY_HOME, 'filtering', 'mask.nii.gz')
    ret = script_runner.run('scil_tractogram_cut_streamlines.py',
                            in_tractogram, '--mask', in_mask,
                            'out_tractogram_cut2.trk',
                            '--resample', '0.2', '--compress', '0.1',
                            '--biggest')
    assert ret.success
