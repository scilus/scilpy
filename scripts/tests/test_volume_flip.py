#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys='surface_vtk_fib.zip')
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_volume_flip.py', '--help')
    assert ret.success


def test_execution_surface_vtk_fib(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fa = os.path.join(get_home(), 'surface_vtk_fib',
                         'fa.nii.gz')
    ret = script_runner.run('scil_volume_flip.py', in_fa, 'fa_flip.nii.gz',
                            'x')
    assert ret.success
