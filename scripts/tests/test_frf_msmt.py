#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_frf_msmt.py', '--help')
    assert ret.success


def test_roi_radii_shape_parameter(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'commit_amico',
                          'dwi.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'commit_amico',
                           'dwi.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'commit_amico',
                           'dwi.bvec')
    mask = os.path.join(SCILPY_HOME, 'commit_amico', 'mask.nii.gz')
    ret = script_runner.run('scil_frf_msmt.py', in_dwi,
                            in_bval, in_bvec, 'wm_frf.txt', 'gm_frf.txt',
                            'csf_frf.txt', '--mask', mask, '--roi_center',
                            '15', '15', '15', '-f')
    assert ret.success

    # Test wrong tolerance, leading to no b0. Current minimal b-val is 5.
    ret = script_runner.run('scil_frf_msmt.py', in_dwi,
                            in_bval, in_bvec, 'wm_frf.txt', 'gm_frf.txt',
                            'csf_frf.txt', '--mask', mask, '--roi_center',
                            '15', '15', '15', '-f', '--tol', '1')
    assert not ret.success

    ret = script_runner.run('scil_frf_msmt.py', in_dwi,
                            in_bval, in_bvec, 'wm_frf.txt', 'gm_frf.txt',
                            'csf_frf.txt', '--mask', mask, '--roi_center',
                            '15', '-f')

    assert (not ret.success)


def test_roi_radii_shape_parameter2(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'commit_amico',
                          'dwi.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'commit_amico',
                           'dwi.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'commit_amico',
                           'dwi.bvec')
    mask = os.path.join(SCILPY_HOME, 'commit_amico', 'mask.nii.gz')
    ret = script_runner.run('scil_frf_msmt.py', in_dwi,
                            in_bval, in_bvec, 'wm_frf.txt', 'gm_frf.txt',
                            'csf_frf.txt', '--mask', mask, '--roi_radii',
                            '37', '-f')
    assert ret.success

    ret = script_runner.run('scil_frf_msmt.py', in_dwi,
                            in_bval, in_bvec, 'wm_frf.txt', 'gm_frf.txt',
                            'csf_frf.txt', '--mask', mask, '--roi_radii',
                            '37', '37', '37', '-f')
    assert ret.success

    ret = script_runner.run('scil_frf_msmt.py', in_dwi,
                            in_bval, in_bvec, 'wm_frf.txt', 'gm_frf.txt',
                            'csf_frf.txt', '--mask', mask, '--roi_radii',
                            '37', '37', '37', '37', '37', '-f')

    assert (not ret.success)


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'commit_amico',
                          'dwi.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'commit_amico',
                           'dwi.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'commit_amico',
                           'dwi.bvec')
    mask = os.path.join(SCILPY_HOME, 'commit_amico', 'mask.nii.gz')
    ret = script_runner.run('scil_frf_msmt.py', in_dwi,
                            in_bval, in_bvec, 'wm_frf.txt', 'gm_frf.txt',
                            'csf_frf.txt', '--mask', mask, '--min_nvox', '20',
                            '-f')
    assert ret.success
