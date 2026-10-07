#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['mrds.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_mrds_metrics', '--help'])
    assert ret.success


def test_execution_mrds_all_metrics(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_evals = os.path.join(SCILPY_HOME,
                            'mrds', 'sub-01_MRDS_eigenvalues.nii.gz')

    # no option
    ret = script_runner.run(['scil_mrds_metrics', in_evals, '-f'])
    assert ret.success


def test_execution_mrds_not_all_metrics(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_evals = os.path.join(SCILPY_HOME,
                            'mrds', 'sub-01_MRDS_eigenvalues.nii.gz')
    in_mask = os.path.join(SCILPY_HOME,
                           'mrds', 'sub-01_mask.nii.gz')
    # no option
    ret = script_runner.run(['scil_mrds_metrics',
                             in_evals,
                             '--mask', in_mask,
                             '--not_all',
                             '--fa', 'sub-01_MRDS_FA.nii.gz',
                             '--ad', 'sub-01_MRDS_AD.nii.gz',
                             '--rd', 'sub-01_MRDS_RD.nii.gz',
                             '--md', 'sub-01_MRDS_MD.nii.gz',
                             '-f'])
    assert ret.success


def test_non_ras_mrds_metrics(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_evals = os.path.join(SCILPY_HOME,
                            'mrds', 'sub-01_MRDS_eigenvalues.nii.gz')
    data = nib.load(in_evals).get_fdata(dtype=np.float32)

    # Save RAS and LAS datasets
    aff_ras = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data, aff_ras), 'evals_ras.nii.gz')

    aff_las = np.diag([-2.0, 2.0, 2.0, 1.0])
    aff_las[0, 3] = 22.0
    data_las = data[::-1, :, :, :].copy()
    nib.save(nib.Nifti1Image(data_las, aff_las), 'evals_las.nii.gz')

    # Run on RAS
    ret_r = script_runner.run(['scil_mrds_metrics', 'evals_ras.nii.gz',
                               '--fa', 'fa_ras.nii.gz',
                               '--md', 'md_ras.nii.gz',
                               '--not_all', '-f'])
    assert ret_r.success

    # Run on LAS
    ret_l = script_runner.run(['scil_mrds_metrics', 'evals_las.nii.gz',
                               '--fa', 'fa_las.nii.gz',
                               '--md', 'md_las.nii.gz',
                               '--not_all', '-f'])
    assert ret_l.success

    # Check orientation was restored and values match
    img_l = nib.load('fa_las.nii.gz')
    assert nib.orientations.aff2axcodes(img_l.affine) == ('L', 'A', 'S')

    fa_r = nib.load('fa_ras.nii.gz').get_fdata()
    fa_l = img_l.get_fdata()
    assert np.allclose(fa_r, fa_l[::-1], atol=1e-5)

    img_md_l = nib.load('md_las.nii.gz')
    assert nib.orientations.aff2axcodes(img_md_l.affine) == ('L', 'A', 'S')

    md_r = nib.load('md_ras.nii.gz').get_fdata()
    md_l = img_md_l.get_fdata()
    assert np.allclose(md_r, md_l[::-1], atol=1e-5)
