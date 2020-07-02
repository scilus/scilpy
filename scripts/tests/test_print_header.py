#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_print_header.py', '--help')
    assert ret.success


def test_execution_img(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(get_home(), 'others',
                          'fa.nii.gz')
    ret = script_runner.run('scil_print_header.py', in_img)
    assert ret.success


def test_execution_tractogram(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto = os.path.join(get_home(), 'others',
                             'IFGWM.trk')
    ret = script_runner.run('scil_print_header.py', in_tracto)
    assert ret.success
