#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_freewater_maps.py', '--help')
    assert ret.success


def test_execution_commit_amico(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'commit_amico',
                          'dwi.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'commit_amico',
                           'dwi.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'commit_amico',
                           'dwi.bvec')
    mask = os.path.join(SCILPY_HOME, 'commit_amico',
                        'mask.nii.gz')
    ret = script_runner.run('scil_freewater_maps.py', in_dwi,
                            in_bval, in_bvec, '--mask', mask,
                            '--out_dir', 'freewater', '--b_thr', '30',
                            '--para_diff', '0.0015',
                            '--perp_diff_min', '0.0001',
                            '--perp_diff_max', '0.0007',
                            '--lambda1', '0.0', '--lambda2', '0.001',
                            '--processes', '1')
    assert ret.success
