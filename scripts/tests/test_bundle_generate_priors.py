#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bst.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_bundle_generate_priors.py', '--help')
    assert ret.success


def test_execution_bst(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'bst', 'rpt_m_lin.trk')
    in_fodf = os.path.join(SCILPY_HOME, 'bst', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'bst', 'mask.nii.gz')
    ret = script_runner.run('scil_bundle_generate_priors.py',
                            in_bundle, in_fodf, in_mask,
                            '--todi_sigma', '1', '--out_prefix', 'rpt_m',
                            '--sh_basis', 'descoteaux07')
    assert ret.success
