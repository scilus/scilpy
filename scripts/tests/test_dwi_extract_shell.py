#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_dwi_extract_shell.py', '--help')
    assert ret.success


def test_execution_processing_1000(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing',
                          'dwi_crop.nii.gz')
    in_bval = os.path.join(get_home(), 'processing',
                           'dwi.bval')
    in_bvec = os.path.join(get_home(), 'processing',
                           'dwi.bvec')
    ret = script_runner.run('scil_dwi_extract_shell.py', in_dwi,
                            in_bval, in_bvec, '0', '1000',
                            'dwi_crop_1000.nii.gz', '1000.bval', '1000.bvec',
                            '-t', '30')
    assert ret.success


def test_execution_out_indices(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing',
                          'dwi_crop.nii.gz')
    in_bval = os.path.join(get_home(), 'processing',
                           'dwi.bval')
    in_bvec = os.path.join(get_home(), 'processing',
                           'dwi.bvec')
    ret = script_runner.run('scil_dwi_extract_shell.py', in_dwi,
                            in_bval, in_bvec, '0', '1000',
                            'dwi_crop_1000__1.nii.gz', '1000__1.bval',
                            '1000__1.bvec', '-t', '30', '--out_indices',
                            'out_indices.txt')
    assert ret.success


def test_execution_processing_3000(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing',
                          'dwi_crop.nii.gz')
    in_bval = os.path.join(get_home(), 'processing',
                           'dwi.bval')
    in_bvec = os.path.join(get_home(), 'processing',
                           'dwi.bvec')
    ret = script_runner.run('scil_dwi_extract_shell.py', in_dwi,
                            in_bval, in_bvec, '0', '3000',
                            'dwi_crop_3000.nii.gz', '3000.bval', '3000.bvec',
                            '-t', '30')
    assert ret.success
