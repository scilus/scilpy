#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys='surface_vtk_fib.zip')
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_surface_apply_transform', '--help'])
    assert ret.success


def test_execution_surface_vtk_fib(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ref = os.path.join(SCILPY_HOME, 'surface_vtk_fib', 'fa.nii.gz')
    in_surf = os.path.join(SCILPY_HOME, 'surface_vtk_fib',
                           'lhpialt.vtk')
    np.savetxt("affine.txt", np.eye(4))
    ret = script_runner.run(['scil_surface_apply_transform', in_surf, ref,
                             "affine.txt", 'lhpialt_lin.vtk', '--inverse',
                             "--reference", ref,
                             '--source_space', 'lpsmm'])
    assert ret.success
