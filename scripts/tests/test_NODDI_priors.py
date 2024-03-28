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
    ret = script_runner.run('scil_NODDI_priors.py', '--help')
    assert ret.success


def test_execution_commit_amico(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fa = os.path.join(SCILPY_HOME, 'processing',
                         'fa.nii.gz')
    in_ad = os.path.join(SCILPY_HOME, 'processing',
                         'ad.nii.gz')
    in_md = os.path.join(SCILPY_HOME, 'processing',
                         'md.nii.gz')
    in_rd = os.path.join(SCILPY_HOME, 'processing',
                         'rd.nii.gz')
    ret = script_runner.run('scil_NODDI_priors.py', in_fa, in_ad, in_rd, in_md,
                            '--out_txt_1fiber_para', '1fiber_para.txt',
                            '--out_txt_1fiber_perp', '1fiber_perp.txt',
                            '--out_mask_1fiber', '1fiber.nii.gz',
                            '--out_txt_ventricles', 'ventricules.txt',
                            '--out_mask_ventricles', 'ventricules.nii.gz')
    assert ret.success
