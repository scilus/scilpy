#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_gradients_validate_correct.py', '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(get_home(), 'processing',
                           '1000.bval')
    in_bvec = os.path.join(get_home(), 'processing',
                           '1000.bvec')

    # generate the peaks file and fa map we'll use to test our script
    script_runner.run('scil_dti_metrics.py', in_dwi, in_bval, in_bvec,
                      '--not_all', '--fa', 'fa.nii.gz',
                      '--evecs', 'evecs.nii.gz')
    # test the actual script
    ret = script_runner.run('scil_gradients_validate_correct.py', in_bvec,
                            'evecs_v1.nii.gz', 'fa.nii.gz', 'bvec_corr', '-v')
    assert ret.success
