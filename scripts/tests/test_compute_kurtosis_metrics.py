#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_kurtosis_metrics.py', '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing',
                          'dwi_crop.nii.gz')
    in_bval = os.path.join(get_home(), 'processing',
                           'dwi.bval')
    in_bvec = os.path.join(get_home(), 'processing',
                           'dwi.bvec')
    ret = script_runner.run('scil_compute_kurtosis_metrics.py', in_dwi,
                            in_bval, in_bvec, '--not_all',
                            '--dki_fa', 'dki_fa.nii.gz',
                            '--dki_md', 'dki_md.nii.gz',
                            '--dki_rd', 'dki_rd.nii.gz',
                            '--dki_ad', 'dki_ad.nii.gz')
    assert ret.success
