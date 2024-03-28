#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_resample.py', '--help')
    assert ret.success


def test_execution_downsample(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto = os.path.join(SCILPY_HOME, 'tracking',
                             'union_shuffle_sub.trk')
    ret = script_runner.run('scil_tractogram_resample.py', in_tracto,
                            '500', 'union_shuffle_sub_downsampled.trk')
    assert ret.success


def test_execution_upsample(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto = os.path.join(SCILPY_HOME, 'tracking',
                             'union_shuffle_sub.trk')

    # in_tracto contains 761 streamlines. 2000=upsample. 200=downsample.

    ret = script_runner.run('scil_tractogram_resample.py', in_tracto,
                            '2000', 'union_shuffle_sub_upsampled.trk', '-f',
                            '--point_wise_std', '0.5')
    assert ret.success

    ret = script_runner.run('scil_tractogram_resample.py', in_tracto,
                            '200', 'union_shuffle_sub_downsampled.trk',
                            '--point_wise_std', '0.5', '-f')
    assert ret.success

    ret = script_runner.run('scil_tractogram_resample.py', in_tracto,
                            '200', 'union_shuffle_sub_downsampled.trk',
                            '--point_wise_std', '0.5', '-f',
                            '--downsample_per_cluster')
    assert ret.success


def test_execution_upsample_ptt(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto = os.path.join(SCILPY_HOME, 'tracking',
                             'union_shuffle_sub.trk')

    ret = script_runner.run('scil_tractogram_resample.py', in_tracto,
                            '500', 'union_shuffle_sub_upsampled.trk', '-f',
                            '--point_wise_std', '10', '--tube_radius', '5')
    assert ret.success
