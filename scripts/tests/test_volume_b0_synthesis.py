#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

tensorflow = pytest.importorskip("tensorflow")


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip', 'processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_volume_b0_synthesis.py', '--help')
    assert ret.success


@pytest.mark.skipif(tensorflow is None, reason="Tensorflow not installed")
def test_synthesis(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_t1 = os.path.join(SCILPY_HOME, 'others',
                         't1.nii.gz')
    in_b0 = os.path.join(SCILPY_HOME, 'processing',
                         'b0_mean.nii.gz')

    t1_img = nib.load(in_t1)
    b0_img = nib.load(in_b0)
    t1_data = t1_img.get_fdata()
    b0_data = b0_img.get_fdata()
    t1_data[t1_data > 0] = 1
    b0_data[b0_data > 0] = 1
    nib.save(nib.Nifti1Image(t1_data.astype(np.uint8), t1_img.affine),
             't1_mask.nii.gz')
    nib.save(nib.Nifti1Image(b0_data.astype(np.uint8), b0_img.affine),
             'b0_mask.nii.gz')

    ret = script_runner.run('scil_volume_b0_synthesis.py',
                            in_t1, 't1_mask.nii.gz',
                            in_b0, 'b0_mask.nii.gz',
                            'b0_synthesized.nii.gz', '-v')
    assert ret.success
