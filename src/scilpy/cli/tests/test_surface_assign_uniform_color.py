#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys='surface_vtk_fib.zip')
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_surface_assign_uniform_color',
                             '--help'])
    assert ret.success


def test_execution_fill(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_surf = os.path.join(SCILPY_HOME, 'surface_vtk_fib', 'lhpialt.vtk')
    ref = os.path.join(SCILPY_HOME, 'surface_vtk_fib', 'fa.nii.gz')

    ret = script_runner.run(['scil_surface_assign_uniform_color',
                             in_surf, '--fill_color', '0x000000',
                             '--out_surface', 'colored.vtk',
                             '--reference', ref,
                             '--source_space', 'lpsmm', '-f'])
    assert ret.success


def test_execution_dict(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_surf = os.path.join(SCILPY_HOME, 'surface_vtk_fib', 'lhpialt.vtk')
    ref = os.path.join(SCILPY_HOME, 'surface_vtk_fib', 'fa.nii.gz')

    # Create a fake dictionary. Using the other hexadecimal format.
    my_dict = {'lhpialt': '#000000'}
    json_file = 'my_json_dict.json'
    with open(json_file, "w+") as f:
        json.dump(my_dict, f)

    ret = script_runner.run(['scil_surface_assign_uniform_color',
                             in_surf, '--dict_colors', json_file,
                             '--out_suffix', 'colored',
                             '--reference', ref,
                             '--source_space', 'lpsmm', '-f'])
    assert ret.success
