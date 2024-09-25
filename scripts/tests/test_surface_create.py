#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['atlas.zip'])
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_surface_create.py', '--help')
    assert ret.success


def test_execution_atlas(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_atlas = os.path.join(SCILPY_HOME, 'atlas',
                            'atlas_freesurfer_v2.nii.gz')
    ret = script_runner.run('scil_surface_create.py',
                            '--in_labels', in_atlas,
                            'surface.vtk',
                            '--list_indices', '2024:2035 1024',
                            '--fill',
                            '--smooth', '1',
                            '--erosion', '1',
                            '--dilation', '1',
                            '--opening', '1',
                            '--closing', '1', '-f')
    assert ret.success


def test_execution_atlas_each_index(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_atlas = os.path.join(SCILPY_HOME, 'atlas',
                            'atlas_freesurfer_v2.nii.gz')
    ret = script_runner.run('scil_surface_create.py',
                            '--in_labels', in_atlas,
                            'surface.vtk',
                            '--each_index',
                            '--fill',
                            '--smooth', '1',
                            '--erosion', '1',
                            '--dilation', '1',
                            '--opening', '1',
                            '--closing', '1',
                            '--vtk2vox', '-f')
    assert ret.success


def test_execution_atlas_no_index(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_atlas = os.path.join(SCILPY_HOME, 'atlas',
                            'atlas_freesurfer_v2.nii.gz')
    ret = script_runner.run('scil_surface_create.py',
                            '--in_labels', in_atlas,
                            'surface.vtk',
                            '--fill',
                            '--smooth', '1',
                            '--erosion', '1',
                            '--dilation', '1',
                            '--opening', '1',
                            '--closing', '1', '-f')
    assert ret.success


def test_execution_mask(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_mask = os.path.join(SCILPY_HOME, 'atlas',
                           'brainstem_bin.nii.gz')
    ret = script_runner.run('scil_surface_create.py',
                            '--in_mask', in_mask,
                            'surface.vtk', '-f')
    assert ret.success


def test_execution_volume(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_t1 = os.path.join(SCILPY_HOME, 'others',
                         't1.nii.gz')
    ret = script_runner.run('scil_surface_create.py',
                            '--in_volume', in_t1,
                            '--value', '0.2',
                            'surface.vtk', '-f')
    assert ret.success
