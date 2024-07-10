#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()

in_tracto_1 = os.path.join(SCILPY_HOME, 'tracking', 'local_split_0.trk')
in_3d_map = os.path.join(SCILPY_HOME, 'tracking', 'fa.nii.gz')
in_4d_map = os.path.join(SCILPY_HOME, 'tracking', 'peaks.nii.gz')


def test_help_option(script_runner):
    ret = script_runner.run(
            'scil_tractogram_project_map_to_streamlines.py', '--help')
    assert ret.success


def test_execution_3D_map(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1, 't1_on_streamlines.trk',
                            '--in_maps', in_3d_map,
                            '--out_dpp_name', 't1')
    assert ret.success


def test_execution_4D_map(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1, 'rgb_on_streamlines.trk',
                            '--in_maps', in_4d_map,
                            '--out_dpp_name', 'rgb')
    assert ret.success


def test_execution_3D_map_endpoints_only(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1,
                            't1_on_streamlines_endpoints.trk',
                            '--in_maps', in_3d_map,
                            '--out_dpp_name', 't1',
                            '--endpoints_only')
    assert ret.success


def test_execution_4D_map_endpoints_only(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1,
                            'rgb_on_streamlines_endpoints.trk',
                            '--in_maps', in_4d_map,
                            '--out_dpp_name', 'rgb',
                            '--endpoints_only')
    assert ret.success


def test_execution_3D_map_trilinear(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1,
                            't1_on_streamlines_trilinear.trk',
                            '--in_maps', in_3d_map,
                            '--out_dpp_name', 't1',
                            '--trilinear')
    assert ret.success
