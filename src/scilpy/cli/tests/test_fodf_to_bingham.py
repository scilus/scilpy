#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from dipy.data import get_sphere
import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.reconst.bingham import bingham_to_sf

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_fodf_to_bingham',
                            '--help'])
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing',
                           'fodf_descoteaux07.nii.gz')
    ret = script_runner.run(['scil_fodf_to_bingham',
                             in_fodf, 'bingham.nii.gz',
                             '--max_lobes', '1',
                             '--at', '0.0',
                             '--rt', '0.1',
                             '--min_sep_angle', '25.',
                             '--max_fit_angle', '15.',
                             '--processes', '1'])
    assert ret.success


def test_execution_processing_mask(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing',
                           'fodf_descoteaux07.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'processing',
                           'seed.nii.gz')
    ret = script_runner.run(['scil_fodf_to_bingham',
                             in_fodf, 'bingham.nii.gz',
                             '--max_lobes', '1',
                             '--at', '0.0',
                             '--rt', '0.1',
                             '--min_sep_angle', '25.',
                             '--max_fit_angle', '15.',
                             '--processes', '1',
                             '--mask', in_mask, '-f'])
    assert ret.success


def test_sh_basis_consistency(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing',
                           'fodf_descoteaux07.nii.gz')

    # Save small patch to keep execution time low.
    data = nib.load(in_fodf).get_fdata(dtype=np.float32)[15:17, 15:17, 15:17]
    nib.save(nib.Nifti1Image(data, np.eye(4)), 'sub_desc.nii.gz')

    # Convert descoteaux07_legacy to tournier07
    ret_conv = script_runner.run(['scil_sh_convert', 'sub_desc.nii.gz',
                                  'sub_tourn.nii.gz',
                                  'descoteaux07_legacy', 'tournier07',
                                  '--processes', '1', '-f'])
    assert ret_conv.success

    # Fit descoteaux07_legacy input
    ret_d = script_runner.run(['scil_fodf_to_bingham', 'sub_desc.nii.gz',
                               'bingham_d.nii.gz',
                               '--sh_basis', 'descoteaux07_legacy',
                               '--max_lobes', '1', '--processes', '1', '-f'])
    assert ret_d.success

    # Fit tournier07 input
    ret_t = script_runner.run(['scil_fodf_to_bingham', 'sub_tourn.nii.gz',
                               'bingham_t.nii.gz',
                               '--sh_basis', 'tournier07',
                               '--max_lobes', '1', '--processes', '1', '-f'])
    assert ret_t.success

    # Reconstructed distributions on the sphere must match across bases
    b_d = nib.load('bingham_d.nii.gz').get_fdata()
    b_t = nib.load('bingham_t.nii.gz').get_fdata()
    sphere = get_sphere(name='repulsion724')
    sf_d = bingham_to_sf(b_d, sphere.vertices)
    sf_t = bingham_to_sf(b_t, sphere.vertices)
    assert np.allclose(sf_d, sf_t, atol=1e-3)
