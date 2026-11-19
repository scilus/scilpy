#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import tempfile

import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()
in_img = os.path.join(SCILPY_HOME, 'processing', 'peaks.nii.gz')


def test_help_option(script_runner):
    ret = script_runner.run(['scil_peak_angular_error', '--help'])
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    out_angular_error = 'peak_angular_error.nii.gz'
    out_json = 'peak_angular_error.json'
    ret = script_runner.run([
        'scil_peak_angular_error', in_img, in_img,
        out_angular_error, out_json])

    assert ret.success

    in_peaks = nib.load(in_img)
    angular_error = nib.load(out_angular_error)

    assert angular_error.shape == in_peaks.shape[:3]
    assert np.allclose(angular_error.affine, in_peaks.affine)
    assert np.isfinite(angular_error.get_fdata()).all()

    with open(out_json, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    assert 'mean_max_angular_error' in metrics
    assert any(key.startswith('mean_max_angular_error_nufo_')
               for key in metrics)
