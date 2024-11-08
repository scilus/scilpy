#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['mrds.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_mrds_metrics.py', '--help')
    assert ret.success


def test_execution_mrds_all_metrics(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_evals = os.path.join(SCILPY_HOME,
                            'mrds', 'sub-01_MRDS_eigenvalues.nii.gz')

    # no option
    ret = script_runner.run('scil_mrds_metrics.py',
                            in_evals,
                            '-f')
    assert ret.success


def test_execution_mrds_not_all_metrics(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_evals = os.path.join(SCILPY_HOME,
                            'mrds', 'sub-01_MRDS_eigenvalues.nii.gz')
    in_mask = os.path.join(SCILPY_HOME,
                           'mrds', 'sub-01_mask.nii.gz')
    # no option
    ret = script_runner.run('scil_mrds_metrics.py',
                            in_evals,
                            '--mask', in_mask,
                            '--not_all',
                            '--fa', 'sub-01_MRDS_FA.nii.gz',
                            '--ad', 'sub-01_MRDS_AD.nii.gz',
                            '--rd', 'sub-01_MRDS_RD.nii.gz',
                            '--md', 'sub-01_MRDS_MD.nii.gz',
                            '-f')
    assert ret.success
