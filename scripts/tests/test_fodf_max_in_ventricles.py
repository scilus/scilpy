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
    ret = script_runner.run('scil_fodf_max_in_ventricles.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing',
                           'fodf.nii.gz')
    in_fa = os.path.join(SCILPY_HOME, 'processing',
                         'fa.nii.gz')
    in_md = os.path.join(SCILPY_HOME, 'processing',
                         'md.nii.gz')
    ret = script_runner.run('scil_fodf_max_in_ventricles.py', in_fodf,
                            '--fa', in_fa, 
                            '--md', in_md,
                            '--sh_basis', 'tournier07')
    assert ret.success


def test_execution_processing_w_mask(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing',
                           'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'processing',
                         'small_roi_gm_mask.nii.gz')
    ret = script_runner.run('scil_fodf_max_in_ventricles.py', in_fodf,
                            '--mask', in_mask, 
                            '--sh_basis', 'tournier07')
    assert ret.success

def test_execution_processing_fail(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing',
                           'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'processing',
                         'small_roi_gm_mask.nii.gz')
    in_fa = os.path.join(SCILPY_HOME, 'processing',
                         'fa.nii.gz')    
    ret = script_runner.run('scil_fodf_max_in_ventricles.py', in_fodf,
                            '--fa', in_fa,
                            '--mask', in_mask, 
                            '--sh_basis', 'tournier07')
    assert not ret.success