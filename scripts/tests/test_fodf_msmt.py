#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_fodf_msmt.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'commit_amico', 'dwi.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'commit_amico', 'dwi.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'commit_amico', 'dwi.bvec')
    in_wm_frf = os.path.join(SCILPY_HOME, 'commit_amico', 'wm_frf.txt')
    in_gm_frf = os.path.join(SCILPY_HOME, 'commit_amico', 'gm_frf.txt')
    in_csf_frf = os.path.join(SCILPY_HOME, 'commit_amico', 'csf_frf.txt')
    mask = os.path.join(SCILPY_HOME, 'commit_amico', 'mask.nii.gz')

    ret = script_runner.run('scil_fodf_msmt.py', in_dwi, in_bval,
                            in_bvec, in_wm_frf, in_gm_frf, in_csf_frf,
                            '--mask', mask,
                            '--wm_out_fODF', 'wm_fodf.nii.gz',
                            '--gm_out_fODF', 'gm_fodf.nii.gz',
                            '--csf_out_fODF', 'csf_fodf.nii.gz',
                            '--vf', 'vf.nii.gz', '--sh_order', '4',
                            '--sh_basis', 'tournier07',
                            '--processes', '1', '-f')
    assert ret.success
