#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys='surface_vtk_fib.zip')
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_surface_convert.py', '--help')
    assert ret.success


def test_execution_surface_vtk_fib(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_surf = os.path.join(SCILPY_HOME, 'surface_vtk_fib',
                           'lhpialt.vtk')
    ret = script_runner.run('scil_surface_convert.py', in_surf,
                            'rhpialt.ply')
    assert ret.success


def test_execution_surface_vtk_xfrom(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_surf = os.path.join(SCILPY_HOME, 'surface_vtk_fib',
                           'lh.pialt_xform')
    x_form = os.path.join(SCILPY_HOME, 'surface_vtk_fib',
                          'log.txt')
    ret = script_runner.run('scil_surface_convert.py', in_surf,
                            'lh.pialt_xform.vtk', '--xform', x_form,
                            '--to_lps')
    assert ret.success
