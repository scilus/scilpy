#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_screenshot_dti.py', '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing',
                          'dwi.nii.gz')
    in_bval = os.path.join(get_home(), 'processing',
                           'dwi.bval')
    in_bvec = os.path.join(get_home(), 'processing',
                           'dwi.bvec')
    in_template = os.path.join(get_home(), 'processing',
                               'mni_masked_2x2x2.nii.gz')
    ret = script_runner.run('scil_screenshot_dti.py', in_dwi, in_bval, in_bvec,
                            in_template, '--shell', '0', '1000',
                            '--out_dir', 'qa/')
    assert ret.success
