#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_dwi_to_sh.py', '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing',
                          'dwi_crop_3000.nii.gz')
    in_bval = os.path.join(get_home(), 'processing',
                           '3000.bval')
    in_bvec = os.path.join(get_home(), 'processing',
                           '3000.bvec')
    ret = script_runner.run('scil_dwi_to_sh.py', in_dwi, in_bval,
                            in_bvec, 'sh_1000.nii.gz')
    assert ret.success
