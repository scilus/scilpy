#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_merge_sh.py', '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_sh_1 = os.path.join(get_home(), 'processing',
                           'sh_1000.nii.gz')
    in_sh_2 = os.path.join(get_home(), 'processing',
                           'sh_3000.nii.gz')
    ret = script_runner.run('scil_merge_sh.py', in_sh_1, in_sh_2, 'sh.nii.gz')
    assert ret.success
