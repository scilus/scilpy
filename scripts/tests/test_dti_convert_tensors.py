#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_dti_convert_tensors.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # No tensor in the current test data! I'm running the dti_metrics
    # to create one.
    in_dwi = os.path.join(SCILPY_HOME, 'processing', 'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing', '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing', '1000.bvec')
    script_runner.run('scil_dti_metrics.py', in_dwi,
                      in_bval, in_bvec, '--not_all',
                      '--tensor', 'tensors.nii.gz', '--tensor_format', 'fsl')

    ret = script_runner.run('scil_dti_convert_tensors.py', 'tensors.nii.gz',
                            'converted_tensors.nii.gz', 'fsl', 'mrtrix')

    assert ret.success
