#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import numpy as np
from dipy.io.surface import load_surface

from dipy.io.utils import Space

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys='surface_vtk_fib.zip')
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_surface_assign_custom_color',
                             '--help'])
    assert ret.success


def test_execution_from_anatomy(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_surf = os.path.join(SCILPY_HOME, 'surface_vtk_fib', 'lhpialt.vtk')
    ref = os.path.join(SCILPY_HOME, 'surface_vtk_fib', 'fa.nii.gz')

    ret = script_runner.run(['scil_surface_assign_custom_color',
                             in_surf, 'colored_anat.vtk',
                             '--from_anatomy', ref,
                             '--reference', ref,
                             '--source_space', 'lpsmm', '-f'])
    assert ret.success


def test_execution_load_dpp(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_surf = os.path.join(SCILPY_HOME, 'surface_vtk_fib', 'lhpialt.vtk')
    ref = os.path.join(SCILPY_HOME, 'surface_vtk_fib', 'fa.nii.gz')

    sfs = load_surface(in_surf, ref, from_space=Space.LPSMM)
    nb_vertices = len(sfs.vertices)
    dpp_data = np.random.rand(nb_vertices)
    np.savetxt('test.txt', dpp_data)

    ret = script_runner.run(['scil_surface_assign_custom_color',
                             in_surf, 'colored_dpp.vtk',
                             '--load_dpp', 'test.txt',
                             '--reference', ref,
                             '--source_space', 'lpsmm', '-f'])
    assert ret.success
