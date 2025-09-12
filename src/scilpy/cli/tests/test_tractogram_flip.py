#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pytest
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys='surface_vtk_fib.zip')
tmp_dir = tempfile.TemporaryDirectory()


@pytest.mark.smoke
def test_help_option(script_runner):
    ret = script_runner.run(['scil_tractogram_flip', '--help'])
    assert ret.success


@pytest.mark.smoke
def test_execution_surface_vtk_fib(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fib = os.path.join(SCILPY_HOME, 'surface_vtk_fib',
                          'gyri_fanning.fib')
    in_fa = os.path.join(SCILPY_HOME, 'surface_vtk_fib',
                         'fa.nii.gz')
    ret = script_runner.run(['scil_tractogram_flip', in_fib,
                             'gyri_fanning.tck', 'x', '--reference', in_fa])
    assert ret.success
