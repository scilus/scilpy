#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_qball_metrics.py', '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(get_home(), 'processing',
                           '1000.bval')
    in_bvec = os.path.join(get_home(), 'processing',
                           '1000.bvec')
    ret = script_runner.run('scil_qball_metrics.py', in_dwi,
                            in_bval, in_bvec)
    assert ret.success


def test_execution_not_all(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(get_home(), 'processing',
                           '1000.bval')
    in_bvec = os.path.join(get_home(), 'processing',
                           '1000.bvec')
    ret = script_runner.run('scil_qball_metrics.py', in_dwi,
                            in_bval, in_bvec, "--not_all", "--sh", "2.nii.gz")
    assert ret.success
