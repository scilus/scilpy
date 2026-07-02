#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tempfile

import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()
in_img = os.path.join(SCILPY_HOME, 'processing', 'peaks.nii.gz')


def test_help_option(script_runner):
    ret = script_runner.run(['scil_peak_to_odf', '--help'])
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # generate temp file
    single_dir_nib = nib.load(in_img).slicer[..., :3]
    single_dir_img = os.path.join(tmp_dir.name, 'single_dir_peaks.nii.gz')
    nib.save(single_dir_nib, single_dir_img)

    out_sh = 'peaks_to_odf.nii.gz'
    ret = script_runner.run(['scil_peak_to_odf', single_dir_img, out_sh])

    assert ret.success

    in_peaks = nib.load(in_img)
    out_peaks = nib.load(out_sh)

    assert out_peaks.shape[:3] == in_peaks.shape[:3]
    assert out_peaks.ndim == 4
    assert np.allclose(out_peaks.affine, in_peaks.affine)
    assert np.isfinite(out_peaks.get_fdata()).all()
